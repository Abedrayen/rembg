import os
from io import BytesIO
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from PIL import Image
import torch
import torchvision.transforms as T
from isnet import ISNetDIS  # Ensure this module is available in your PYTHONPATH

# Initialize Flask app
app = Flask(__name__)

# Device configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ISNET_MODEL_PATH = "isnet-general-use.pth"

# Initialize and load the model
isnet = ISNetDIS(in_ch=3, out_ch=1)

def load_model(model, model_path, device):
    """
    Load model weights and set the model to evaluation mode.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

isnet = load_model(model=isnet, model_path=ISNET_MODEL_PATH, device=DEVICE)

# Define image transformations
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
    Normalize the predicted map to [0, 1].
    """
    mi = np.min(predicted_map)
    ma = np.max(predicted_map)
    return (predicted_map - mi) / (ma - mi + 1e-5)

def generate_mask(image, blend_weight=0.2):
    """
    Generate a mask from the input image with edge enhancement.
    
    Parameters:
        image (PIL.Image): Input image.
        blend_weight (float): Weight of the edge map to blend with the raw prediction.
    
    Returns:
        numpy.ndarray: Final mask in [0, 255] format.
    """
    # Preprocess and get prediction
    image_tensor = transforms_pipeline(image).to(DEVICE)
    with torch.no_grad():
        result = isnet(image_tensor.unsqueeze(0))
    
    pred_map = norm_pred(result[0][0].cpu().numpy())
    width, height = image.size
    pred_resized = cv2.resize(np.squeeze(pred_map), (width, height), interpolation=cv2.INTER_LINEAR)

    # Compute Canny edges
    canny_edges = cv2.Canny((pred_resized * 255).astype(np.uint8), 50, 200)
    edges_normalized = canny_edges.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Blend the original prediction with the edge map
    combined_mask = cv2.addWeighted(pred_resized, 1.0, edges_normalized, blend_weight, 0)

    # Optional: refine the neck area
    h, w = combined_mask.shape
    neck_top, neck_bottom = int(h * 0.4), int(h * 0.6)
    roi = combined_mask[neck_top:neck_bottom, :]
    roi = cv2.GaussianBlur(roi, (1, 1), 0)
    roi = cv2.addWeighted(roi, 1.5, roi, -0.5, 0)
    combined_mask[neck_top:neck_bottom, :] = roi

    # Clip and scale the final mask to 0-255
    combined_mask = np.clip(combined_mask, 0, 1)
    return (combined_mask * 255).astype(np.uint8)

def remove_background(image, blend_weight=0.2):
    """
    Remove the background from the image using the predicted mask.
    
    Parameters:
        image (PIL.Image): Input image.
        blend_weight (float): Weight for blending the edge map.
    
    Returns:
        PIL.Image: Image with background removed.
    """
    width, height = image.size
    image_tensor = transforms_pipeline(image).to(DEVICE)
    with torch.no_grad():
        result = isnet(image_tensor.unsqueeze(0))
    
    pred_map = norm_pred(result[0][0].cpu().numpy())
    pred_resized = cv2.resize(np.squeeze(pred_map), (width, height), interpolation=cv2.INTER_LINEAR)

    # Compute Canny edges and blend
    canny_edges = cv2.Canny((pred_resized * 255).astype(np.uint8), 50, 200)
    edges_normalized = canny_edges.astype(np.float32) / 255.0  # Normalize to [0, 1]
    combined_mask = cv2.addWeighted(pred_resized, 1.0, edges_normalized, blend_weight, 0)

    # Refine neck area
    h, w = combined_mask.shape
    neck_top, neck_bottom = int(h * 0.5), int(h * 0.6)
    roi = combined_mask[neck_top:neck_bottom, :]
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    roi = cv2.addWeighted(roi, 1.5, roi, -0.5, 0)
    combined_mask[neck_top:neck_bottom, :] = roi
    
    # Create alpha matte and composite with original image
    alpha = np.clip(combined_mask, 0, 1)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    alpha = cv2.erode(alpha, kernel1, iterations=7, borderType=cv2.BORDER_REFLECT)
    alpha = cv2.dilate(alpha, kernel2, iterations=1, borderType=cv2.BORDER_REFLECT)

    image_np = np.array(image)
    foreground = (image_np * alpha[:, :, None]).astype(np.uint8)
    return Image.fromarray(foreground)

def generate_trimap(image, fg_threshold=0.9, bg_threshold=0.1, kernel_size=5):
    """
    Generate a trimap from the predicted probability mask using two thresholds.
    
    Parameters:
        image (PIL.Image): Input image.
        fg_threshold (float): Confidence threshold for definite foreground (range 0 to 1).
                              Pixels with a normalized mask value above this are marked as foreground.
        bg_threshold (float): Confidence threshold for definite background (range 0 to 1).
                              Pixels with a normalized mask value below this are marked as background.
        kernel_size (int): Size of the structuring element used for morphological erosion
                           to refine the definite regions.
                           
    Returns:
        numpy.ndarray: A trimap image with:
                       - 255 for certain foreground,
                       - 0 for certain background,
                       - 128 for unknown regions.
    """
    # Get the predicted mask from your model (mask values are expected in 0-255)
    mask = generate_mask(image)
    
    # Convert mask to a probability map in the range [0, 1]
    mask_prob = mask.astype(np.float32) / 255.0

    # Create binary maps for definite foreground and definite background
    fg_mask = (mask_prob >= fg_threshold).astype(np.uint8) * 255  # definite foreground
    bg_mask = (mask_prob <= bg_threshold).astype(np.uint8) * 255  # definite background

    # Use morphological erosion to refine these masks and remove noise along the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    bg_mask = cv2.erode(bg_mask, kernel, iterations=1)

    # Initialize the trimap with the unknown value (128)
    trimap = np.full(mask.shape, 128, dtype=np.uint8)
    
    # Mark the definite foreground regions with 255
    trimap[fg_mask == 255] = 255
    
    # Mark the definite background regions with 0
    trimap[bg_mask == 255] = 0

    return trimap


# Limit file upload size (e.g., 200 MB)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# API endpoint to get the trimap
@app.route('/get_trimap', methods=['POST'])
def get_trimap():
    """
    API endpoint to generate a trimap from an uploaded image.
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
    Endpoint to generate and return a mask image.
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
    Endpoint to remove the background from an image.
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
    elif extension == 'jpg':
        extension = 'JPEG'
    else:
        extension = extension.upper()

    try:
        result_image = remove_background(image)
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

    img_buffer = BytesIO()
    result_image.save(img_buffer, format=extension,**metadata)
    img_buffer.seek(0)
    mimetype = f'image/{extension.lower()}'
    
    return send_file(img_buffer, mimetype=mimetype,
                     as_attachment=True, download_name=f"bg_removed.{extension.lower()}")

if __name__ == '__main__':
    app.run(debug=True)
