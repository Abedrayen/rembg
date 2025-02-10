import os
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from PIL import Image
import torch
import torchvision.transforms as T

# Custom modules – ensure these modules are in your PYTHONPATH.
from dataloader import read_image, read_trimap
from networks.models import build_model  # 'pred' should be available here.
from isnet import ISNetDIS  # Ensure this module is available in your PYTHONPATH
from demo import np_to_torch, pred, scale_input


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB upload limit
cuda_available = torch.cuda.is_available()
print("Is CUDA available:", cuda_available)

# Optionally, print additional CUDA information if available
if cuda_available:
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current CUDA Device:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
ISNET_MODEL_PATH = "isnet-general-use.pth"
FBA_WEIGHTS_PATH = "FBA.pth"  # Ensure this file exists

# Image transformation constants
MEAN = torch.tensor([0.6, 0.6, 0.6])
STD = torch.tensor([0.55, 0.55, 0.55])
RESIZE_SHAPE = (1024, 1024)

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------

def load_isnet_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load the ISNet model weights and set the model to evaluation mode.
    """
    model = ISNetDIS(in_ch=3, out_ch=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load the ISNet model for mask generation.
isnet_model = load_isnet_model(ISNET_MODEL_PATH, DEVICE)

# -----------------------------------------------------------------------------
# Image Transform Pipeline
# -----------------------------------------------------------------------------

transforms_pipeline = T.Compose([
    T.Resize(RESIZE_SHAPE),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def norm_pred(predicted_map: np.ndarray) -> np.ndarray:
    """
    Normalize the predicted map to the range [0, 1].
    """
    mi = np.min(predicted_map)
    ma = np.max(predicted_map)
    return (predicted_map - mi) / (ma - mi + 1e-5)


def generate_mask(image: Image.Image, blend_weight: float = 0.2) -> np.ndarray:
    """
    Generate a mask from the input image with edge enhancement.
    
    Parameters:
        image (PIL.Image.Image): Input image.
        blend_weight (float): Weight of the edge map to blend with the raw prediction.
    
    Returns:
        numpy.ndarray: Final mask in [0, 255] format.
    """
    # Preprocess and get prediction
    image_tensor = transforms_pipeline(image).to(DEVICE)
    with torch.no_grad():
        result = isnet_model(image_tensor.unsqueeze(0))
    
    pred_map = norm_pred(result[0][0].cpu().numpy())
    
    # Debug: Check predicted mask shape
    if pred_map.size == 0:
        raise ValueError("The predicted mask is empty.")
    print("Predicted mask shape:", pred_map.shape)
    
    # Get original image dimensions and verify they are valid
    width, height = image.size
    if width == 0 or height == 0:
        raise ValueError("The input image has zero width or height.")
    print("Original image size:", (width, height))
    
    # Resize predicted mask to the original image size
    pred_resized = cv2.resize(np.squeeze(pred_map), (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Compute Canny edges
    canny_edges = cv2.Canny((pred_resized * 255).astype(np.uint8), 50, 200)
    edges_normalized = canny_edges.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Blend the original prediction with the edge map
    combined_mask = cv2.addWeighted(pred_resized, 1.0, edges_normalized, blend_weight, 0)
    
    # Optional: refine a region (e.g., neck area)
    h, w = combined_mask.shape
    neck_top, neck_bottom = int(h * 0.4), int(h * 0.6)
    roi = combined_mask[neck_top:neck_bottom, :]
    roi = cv2.GaussianBlur(roi, (1, 1), 0)
    roi = cv2.addWeighted(roi, 1.5, roi, -0.5, 0)
    combined_mask[neck_top:neck_bottom, :] = roi
    
    # Clip and scale the final mask to 0-255
    combined_mask = np.clip(combined_mask, 0, 1)
    return (combined_mask * 255).astype(np.uint8)


def generate_trimap(image: Image.Image,
                      fg_threshold: float = 0.9,
                      bg_threshold: float = 0.1,
                      kernel_size: int = 5) -> np.ndarray:
    """
    Generate a trimap from the predicted mask using foreground and background thresholds.
    
    Parameters:
        image (PIL.Image.Image): Input image.
        fg_threshold (float): Threshold for definite foreground.
        bg_threshold (float): Threshold for definite background.
        kernel_size (int): Size of the structuring element for morphological erosion.
    
    Returns:
        numpy.ndarray: Trimap image with values:
                       - 255 for definite foreground,
                       - 0 for definite background,
                       - 128 for unknown regions.
    """
    mask = generate_mask(image)
    mask_prob = mask.astype(np.float32) / 255.0

    # Create binary maps for definite foreground and background.
    fg_mask = (mask_prob >= fg_threshold).astype(np.uint8) * 255
    bg_mask = (mask_prob <= bg_threshold).astype(np.uint8) * 255

    # Use morphological erosion to refine the definite regions.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    bg_mask = cv2.erode(bg_mask, kernel, iterations=1)

    # Initialize trimap with unknown value (128) and update known regions.
    trimap = np.full(mask.shape, 128, dtype=np.uint8)
    trimap[fg_mask == 255] = 255
    trimap[bg_mask == 255] = 0

    return trimap


class FBAArgs:
    """
    Arguments for the FBA matting model.
    """
    def __init__(self, encoder: str = 'resnet50_GN_WS', decoder: str = 'fba_decoder', weights: str = FBA_WEIGHTS_PATH):
        self.encoder = encoder
        self.decoder = decoder
        self.weights = weights


fba_args = FBAArgs()


def remove_background(image: Image.Image, blend_weight: float = 0.2, device: torch.device = DEVICE) -> Image.Image:
    """
    Remove the background from an image using a matting model.
    
    This function uses a trimap generated from the image and a secondary matting
    model (e.g. FBA matting) to extract the foreground.
    
    Parameters:
        image (PIL.Image.Image): Input image.
        blend_weight (float): Parameter for blending (if applicable).
        device (torch.device): Device on which to run the model.
    
    Returns:
        PIL.Image.Image: Image with the background removed.
    """
    # Convert the PIL image to a NumPy array.
    image_np = np.array(image)

    # Generate a trimap from the image.
    trimap = generate_trimap(image)

    # Build and load the FBA matting model.
    model = build_model(fba_args.weights).to(device)
    model.eval()

    # Predict foreground, background, and the alpha matte.
    # Note: The `pred` function must be defined and imported correctly.
    fg, bg, alpha = pred(image_np, trimap, model)

    # Blend the original image with the predicted alpha matte.
    foreground = (image_np * alpha[:, :, None]).astype(np.uint8)
    return Image.fromarray(foreground)

# -----------------------------------------------------------------------------
# Flask API Endpoints
# -----------------------------------------------------------------------------

@app.route('/get_trimap', methods=['POST'])
def get_trimap_endpoint():
    """
    Generate a trimap from an uploaded image.
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

    return send_file(BytesIO(buffer.tobytes()),
                     mimetype='image/png',
                     as_attachment=True,
                     download_name="generated_trimap.png")


@app.route('/get_mask', methods=['POST'])
def get_mask_endpoint():
    """
    Generate and return a mask image.
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

    return send_file(BytesIO(buffer.tobytes()),
                     mimetype='image/png',
                     as_attachment=True,
                     download_name="generated_mask.png")


@app.route('/remove_background', methods=['POST'])
def remove_background_endpoint():
    """
    Remove the background from an uploaded image.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    # Preserve original metadata and determine output image format.
    metadata = image.info
    extension = file.filename.split('.')[-1].lower()
    if extension in ['tif', 'tiff']:
        extension = 'TIFF'
    elif extension in ['jpg', 'jpeg']:
        extension = 'JPEG'
    else:
        extension = extension.upper()

    try:
        result_image = remove_background(image)
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

    img_buffer = BytesIO()
    result_image.save(img_buffer, format=extension, **metadata)
    img_buffer.seek(0)
    mimetype = f'image/{extension.lower()}'
    
    return send_file(img_buffer,
                     mimetype=mimetype,
                     as_attachment=True,
                     download_name=f"bg_removed.{extension.lower()}")

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
