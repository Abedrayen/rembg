import os
from io import BytesIO
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from flask import Flask, request, send_file, jsonify
from PIL import Image

# ------------------------------
# Import ISNet and FBA modules
# ------------------------------
from isnet import ISNetDIS  # Ensure your ISNet module is available

# FBA imports from your notebook code
from demo import pred  # This function performs FBA matting inference
from networks.models import build_model

# ------------------------------
# Initialize Flask app
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Device configuration
# ------------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load ISNet model
# ------------------------------
ISNET_MODEL_PATH = "isnet-general-use.pth"

def load_model(model, model_path, device):
    """
    Load model weights and set the model to evaluation mode.
    """
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model

isnet = ISNetDIS(in_ch=3, out_ch=1)
isnet = load_model(isnet, ISNET_MODEL_PATH, DEVICE)

# ------------------------------
# Build the FBA matting model
# ------------------------------
class FBAArgs:
    encoder = 'resnet50_GN_WS'
    decoder = 'fba_decoder'
    weights = 'FBA.pth'
    
fba_args = FBAArgs()
try:
    fba_model = build_model(fba_args.weights)
except Exception as e:
    # If the model fails to build (e.g. weights not found), you may add code to download it.
    raise RuntimeError("Failed to build the FBA model. Ensure that 'FBA.pth' is available.") from e
fba_model.eval()  # set FBA model to evaluation mode

# ------------------------------
# ISNet Transformation Pipeline
# ------------------------------
MEAN = torch.tensor([0.6, 0.6, 0.6])
STD = torch.tensor([0.55, 0.55, 0.55])
RESIZE_SHAPE = (1024, 1024)
transforms_pipeline = T.Compose([
    T.Resize(RESIZE_SHAPE),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

def norm_pred(predicted_map):
    """
    Normalize a predicted map to the range [0,1].
    """
    mi = np.min(predicted_map)
    ma = np.max(predicted_map)
    return (predicted_map - mi) / (ma - mi + 1e-5)

# ------------------------------
# ISNet-based Mask and Trimap Generation
# ------------------------------
def generate_mask(image, blend_weight=0.2):
    """
    Generate a coarse mask (values 0-255) from the input image using ISNet
    with some edge enhancement.
    
    Parameters:
        image (PIL.Image): Input image.
        blend_weight (float): Weight for blending an edge map.
    
    Returns:
        numpy.ndarray: Coarse mask (uint8, 0-255).
    """
    # Preprocess and predict with ISNet
    image_tensor = transforms_pipeline(image).to(DEVICE)
    with torch.no_grad():
        result = isnet(image_tensor.unsqueeze(0))
    pred_map = norm_pred(result[0][0].cpu().numpy())
    
    width, height = image.size
    pred_resized = cv2.resize(np.squeeze(pred_map), (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Add Canny edge details
    canny_edges = cv2.Canny((pred_resized * 255).astype(np.uint8), 50, 200)
    edges_normalized = canny_edges.astype(np.float32) / 255.0
    combined_mask = cv2.addWeighted(pred_resized, 1.0, edges_normalized, blend_weight, 0)
    
    # Optional: refine a region (e.g. neck area)
    h, w = combined_mask.shape
    neck_top, neck_bottom = int(h * 0.4), int(h * 0.6)
    roi = combined_mask[neck_top:neck_bottom, :]
    roi = cv2.GaussianBlur(roi, (1, 1), 0)
    roi = cv2.addWeighted(roi, 1.5, roi, -0.5, 0)
    combined_mask[neck_top:neck_bottom, :] = roi
    
    combined_mask = np.clip(combined_mask, 0, 1)
    return (combined_mask * 255).astype(np.uint8)

def generate_trimap(image, fg_threshold=0.9, bg_threshold=0.1, kernel_size=5):
    """
    Generate a trimap from the ISNet-based coarse mask.
    
    Definite foreground pixels (mask >= fg_threshold) are set to 255,
    definite background (mask <= bg_threshold) are set to 0, and the unknown
    region is set to 128.
    
    Returns:
        numpy.ndarray: Single-channel trimap (uint8, values 0, 128, or 255).
    """
    mask = generate_mask(image)
    mask_prob = mask.astype(np.float32) / 255.0
    fg_mask = (mask_prob >= fg_threshold).astype(np.uint8) * 255
    bg_mask = (mask_prob <= bg_threshold).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    bg_mask = cv2.erode(bg_mask, kernel, iterations=1)
    
    trimap = np.full(mask.shape, 128, dtype=np.uint8)
    trimap[fg_mask == 255] = 255
    trimap[bg_mask == 255] = 0
    return trimap

# ------------------------------
# Utility functions for FBA integration
# ------------------------------
def pil_to_np(image):
    """
    Convert a PIL image to a NumPy array in [0,1] range (float32).
    """
    return np.array(image).astype(np.float32) / 255.0

def convert_trimap_to_two_channel(trimap_single):
    """
    Convert a single-channel trimap (with values 0, 0.5, 1) into a two-channel
    representation. For definite background, set channel 0 to 1; for definite foreground,
    set channel 1 to 1; and for unknown regions both channels remain 0.
    
    Parameters:
        trimap_single (numpy.ndarray): 2D array with values in [0,1].
        
    Returns:
        numpy.ndarray: Two-channel trimap of shape (H, W, 2) (float32).
    """
    H, W = trimap_single.shape
    trimap_two = np.zeros((H, W, 2), dtype=np.float32)
    fg_mask = trimap_single >= 0.99   # definite foreground
    bg_mask = trimap_single <= 0.01   # definite background
    trimap_two[bg_mask, 0] = 1.0
    trimap_two[fg_mask, 1] = 1.0
    return trimap_two

# ------------------------------
# Background Removal using ISNet + FBA
# ------------------------------
def remove_background_with_fba(image):
    """
    Remove the background from an image using a two-stage process:
      1. Use ISNet to generate a coarse trimap.
      2. Refine the alpha matte (and compute the foreground) using FBA matting.
    
    Parameters:
        image (PIL.Image): Input image.
    
    Returns:
        PIL.Image: The composite image (foreground multiplied by the refined alpha matte).
    """
    # Convert image to numpy (range [0,1]) as expected by FBA.
    image_np = pil_to_np(image)
    
    # Generate a coarse trimap from ISNet.
    # (This returns a single-channel image with values 0 (bg), 128 (unknown), 255 (fg).)
    coarse_trimap = generate_trimap(image)
    coarse_trimap_norm = coarse_trimap.astype(np.float32) / 255.0  # Values ~0, 0.5, 1
    
    # Convert the single-channel trimap to a two-channel representation.
    trimap_two = convert_trimap_to_two_channel(coarse_trimap_norm)
    
    # Compress the two-channel trimap back to a single channel as in the original notebook.
    trimap_for_fba = trimap_two
    print("trimap_for_fba shape:", trimap_for_fba.shape)  # Should output (H, W)
    
    # Use FBA's prediction function to get foreground, background, and alpha.
    fg, bg, alpha = pred(image_np, trimap_for_fba, fba_model)
    
    # Composite the refined foreground: multiply by the refined alpha matte.
    composite = fg * alpha[..., np.newaxis]
    composite = (composite * 255).astype(np.uint8)
    return Image.fromarray(composite)

# ------------------------------
# Flask Endpoints
# ------------------------------

# Limit file upload size (e.g., 200 MB)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

@app.route('/get_trimap', methods=['POST'])
def get_trimap():
    """
    API endpoint to generate and return a trimap (from ISNet).
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    trimap = generate_trimap(image)
    success, buffer = cv2.imencode('.png', trimap)
    if not success:
        return jsonify({"error": "Failed to encode trimap image."}), 500

    trimap_bytes = BytesIO(buffer.tobytes())
    return send_file(trimap_bytes, mimetype='image/png',
                     as_attachment=True, download_name="generated_trimap.png")

@app.route('/get_mask', methods=['POST'])
def get_mask():
    """
    Endpoint to generate and return the coarse mask (from ISNet).
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file in the request"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    mask = generate_mask(image)
    success, buffer = cv2.imencode('.png', mask)
    if not success:
        return jsonify({"error": "Failed to encode mask image."}), 500

    mask_bytes = BytesIO(buffer.tobytes())
    return send_file(mask_bytes, mimetype='image/png',
                     as_attachment=True, download_name="generated_mask.png")

@app.route('/remove_background', methods=['POST'])
def handle_background_removal():
    """
    API endpoint to remove the background using ISNet + FBA matting.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    # Preserve original metadata and format
    metadata = image.info
    extension = file.filename.split('.')[-1].lower()
    if extension in ['tif', 'tiff']:
        extension = 'TIFF'
    elif extension in ['jpg', 'jpeg']:
        extension = 'JPEG'
    else:
        extension = extension.upper()

    try:
        result_image = remove_background_with_fba(image)
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

    img_buffer = BytesIO()
    result_image.save(img_buffer, format=extension, **metadata)
    img_buffer.seek(0)
    mimetype = f'image/{extension.lower()}'
    
    return send_file(img_buffer, mimetype=mimetype,
                     as_attachment=True, download_name=f"bg_removed.{extension.lower()}")

if __name__ == '__main__':
    app.run(debug=True)
