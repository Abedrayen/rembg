from flask import Flask, request, send_file, jsonify
import os
from PIL import Image
import torch
import torchvision.transforms as T
from isnet import ISNetDIS
import numpy as np
from io import BytesIO
import cv2

# Initialize the Flask application
app = Flask(__name__)

# Determine whether to use GPU or CPU for computations
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define the path to the ISNet model weights
ISNET_MODEL_PATH = "isnet-general-use.pth"

# Initialize the ISNet model
isnet = ISNetDIS(in_ch=3, out_ch=1)

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

isnet = load_model(model=isnet, model_path=ISNET_MODEL_PATH, device=DEVICE)

MEAN = torch.tensor([0.45, 0.45, 0.45])
STD = torch.tensor([0.5, 0.5, 0.5])
resize_shape = (2048, 2048)
transforms = T.Compose([T.Resize(resize_shape), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


def normPRED(predicted_map):
    ma = np.max(predicted_map)
    mi = np.min(predicted_map)
    return (predicted_map - mi) / (ma - mi + 1e-5)

def post_process_mask(mask):
    # mask_uint8 = (mask * 255).astype(np.uint8)
    # # mask_eq = cv2.equalizeHist(mask_uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # mask_morph = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    # mask_blur = cv2.GaussianBlur(mask_morph, (3,3), 0)
    # return mask_blur / 255.0
# this code give me a better mask for some images like the hand and the nick with the vest jean 
    mask = (mask * 255).astype(np.uint8)
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    mask = clahe.apply(mask)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)# Morphological closing to fill small gaps
    mask = cv2.bilateralFilter(mask,  d=9, sigmaColor=75, sigmaSpace=75) # Gaussian Blur for smooth edges
    laplacian = cv2.Laplacian(mask, cv2.CV_64F)# Apply Laplacian Edge Enhancement
    laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)# Convert Laplacian to uint8 before adding
    mask = cv2.addWeighted(mask,0.8, laplacian, 0.,0)   # Add Laplacian to mask safely
    # Normalize back to [0, 1]
    return (mask / 255.0).astype(np.float32)


def generate_mask(image):
    image_trans = transforms(image).to(DEVICE)
    with torch.no_grad():
        result = isnet(image_trans.unsqueeze(0))
    pred_normalize = normPRED(result[0][0].cpu().numpy())
    pred_final_resized = cv2.resize(np.squeeze(pred_normalize), image.size, interpolation=cv2.INTER_LINEAR)
    refined_mask = post_process_mask(pred_final_resized)
    return (refined_mask * 255).astype(np.uint8)

def remove_background(image, mask):
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    mask = cv2.GaussianBlur(normPRED(mask), (5, 5), 0)
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    foreground = image_np * mask_3ch
    background = np.zeros_like(image_np)
    final_image = np.where(mask_3ch > 0.5, foreground, background).astype(np.uint8)
    return Image.fromarray(final_image)
app.config['MAX_CONTENT_LENGTH']=200*1024*1024
@app.route('/get_mask', methods=['POST'])
def get_mask():
    if 'image' not in request.files:
        return jsonify({"error": "No image file in the request"}), 400
    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400
    mask = generate_mask(image)
    _, buffer = cv2.imencode('.png', mask)
    mask_bytes = BytesIO(buffer)
    return send_file(mask_bytes, mimetype='image/png', as_attachment=True, download_name="generated_mask.png")

@app.route('/remove_background', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return jsonify({"error": "No image file in the request"}), 400
    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400
    
      # Extract metadata and other properties from the image
    metadata = image.info  # Get metadata (e.g., EXIF data)
    original_size = image.size  # Save the original size of the image
    original_extension = file.filename.split('.')[-1].lower()  # Extract the file extension
    if original_extension == 'tif':  # Handle special cases for extensions
        original_extension = 'TIFF'
    elif original_extension == 'jpg':
        original_extension = 'JPEG'
   
    # Apply the defined transformations to the image
    image_trans = transforms(image).to(DEVICE)
    with torch.no_grad():  # Disable gradient computation for efficiency
        result = isnet(image_trans.unsqueeze(0))  # Run the model on the input image
    
    # Normalize and resize the predicted mask
    pred_normalize = normPRED(result[0][0].cpu().numpy())  # Normalize the mask
    pred_normalize_squeezed = np.squeeze(pred_normalize)  # Remove extra dimensions
    pred_final_resized = cv2.resize(pred_normalize_squeezed, original_size, interpolation=cv2.INTER_LINEAR)  # Resize to original size
    
    mask_uint8 = (pred_final_resized * 255).astype(np.uint8)

    # # Combine edges and the mask
    pred_final_resized = pred_final_resized.astype(np.float32)
  
    # Apply Laplacian edge detection
    laplacian_edges = cv2.Laplacian(pred_final_resized, cv2.CV_32F)  # Use float32 format
    laplacian_edges = np.abs(laplacian_edges)  # Take absolute values of the Laplacian
    laplacian_edges_normalized = cv2.normalize(laplacian_edges, None, 0, 1, cv2.NORM_MINMAX)  # Normalize to [0, 1]

   # Détection de la zone du cou (approche basée sur le masque)
    height, width = pred_final_resized.shape
    region_of_interest = pred_final_resized[int(height * 0.4):int(height * 0.6), :]  # Ajuster ces proportions pour cibler le cou

    # Appliquer un éclairage ou une modification localisée
    roi_brightness = cv2.GaussianBlur(region_of_interest, (7, 7), 0)  # Doucement flouter
    roi_brightness = cv2.addWeighted(roi_brightness, 1.5, region_of_interest, -0.5, 0)  # Ajustement d'éclairage

    # Réintégrer la modification dans l'image principale
    pred_final_resized[int(height * 0.4):int(height * 0.6), :] = roi_brightness
    alpha = np.clip(pred_final_resized, 0, 1)
    
    
    image_np = np.array(image)
    foreground = (image_np * alpha[:, :, None]).astype(np.uint8)
    background = np.zeros_like(image_np)
    final_image = cv2.addWeighted(foreground, 1, background, 0, 0)
    
    # Save the final result into an in-memory binary stream
    img_io = BytesIO()
    output_image = Image.fromarray(final_image)  # Convert to PIL image
    output_image.save(img_io, format=original_extension, **metadata)  # Save with metadata
    img_io.seek(0)  # Reset stream position to the start
    
    # Return the image as a downloadable file
    return send_file(
        img_io,
        mimetype=f'image/{original_extension.lower()}',  # Set MIME type based on file extension
        as_attachment=True,  # Force download
        download_name=f"background_removed.{original_extension.lower()}"  # Set download file name
    )
if __name__ == '__main__':
    app.run(debug=True)
