from flask import Flask, request, send_file, jsonify  # Import Flask for the web framework and other utilities
import os  # For interacting with the operating system
from PIL import Image  # For image processing
import torch  # For PyTorch operations
import torchvision.transforms as T  # For image transformations
from isnet import ISNetDIS  # Importing the ISNet model
import numpy as np  # For numerical operations
from io import BytesIO  # For handling in-memory binary streams
import cv2  # For image processing tasks

# Initialize the Flask application
app = Flask(__name__)

# Determine whether to use GPU or CPU for computations
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define the path to the ISNet model weights
ISNET_MODEL_PATH = "isnet-general-use.pth"

# Initialize the ISNet model
isnet = ISNetDIS(in_ch=3, out_ch=1)

# Function to load a pre-trained model
def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model weights
    model = model.to(device)  # Move the model to the specified device (CPU/GPU)
    model.eval()  # Set the model to evaluation mode
    return model

# Load the ISNet model into memory
isnet = load_model(model=isnet, model_path=ISNET_MODEL_PATH, device=DEVICE)

MEAN = torch.tensor([0.45, 0.45, 0.45]) #(adjust this mean is the last thing before conbine another pretrained model to this one to detect perfectly the contours )
STD = torch.tensor([0.5,0.5,0.5])
resize_shape = (2048, 2048)  # Larger resize shape for better detail capture  (the last thing also )
transforms = T.Compose([T.Resize(resize_shape), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

# Normalize a predicted mask to the range [0, 1]
def normPRED(predicted_map):
    ma = np.max(predicted_map)  # Find the maximum value in the map
    mi = np.min(predicted_map)  # Find the minimum value in the map
    map_normalize = (predicted_map - mi) / (ma - mi + 1e-5)  # Normalize the map to [0, 1]
    return map_normalize

# Remove the background from an image using the predicted mask
def remove_background(image, mask):
    image_np = np.array(image)  # Convert the PIL image to a NumPy array
    if len(image_np.shape) == 2:  # If the image is grayscale
        image_np = np.stack([image_np] * 3, axis=-1)  # Convert it to RGB by stacking
    
    # Smooth the mask with Gaussian blur
    mask = cv2.GaussianBlur(normPRED(mask), (5, 5), 0) #(7,7)
    
    # Expand the mask to have 3 channels
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    foreground = image_np * mask_3ch  # Apply the mask to the image
    background = np.zeros_like(image_np)  # Create a black background
    final_image = np.where(mask_3ch > 0.5, foreground, background).astype(np.uint8)  # Combine the mask and background
    
    return Image.fromarray(final_image)
app.config['MAX_CONTENT_LENGTH']=200*1024*1024
@app.route('/remove_background', methods=['POST'])

def remove_bg():
    if 'image' not in request.files:  # Check if an image file is included in the request
        return jsonify({"error": "No image file in the request"}), 400

    file = request.files['image']  # Get the image file from the request
    try:
        # Open the image and ensure it's in RGB format
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:  # Handle invalid image errors
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
    
    # median_val = np.median(mask_uint8)
    # lower_thresh = int(max(0, 0.66 * median_val))
    # upper_thresh = int(min(255, 1.33 * median_val))
    # edges = cv2.Canny(mask_uint8, 20, 100)
    
    # edges_dilated = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=2)
    # edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8), iterations=2)
    # edges_closed  = (edges_closed / 255.0).astype(np.float32)
    
    # # Combine edges and the mask
    pred_final_resized = pred_final_resized.astype(np.float32)
    # pred_final_resized = cv2.add(pred_final_resized, edges_closed )
    # pred_final_resized = np.clip(pred_final_resized, 0, 1)
    
     # Smooth the mask using GaussianBlur
    # blurred = cv2.GaussianBlur(pred_final_resized, (5, 5), 0)  # Kernel size can be adjusted
    
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

    
    # border_mask = cv2.Canny((pred_final_resized * 255).astype(np.uint8), 50, 150)
    # border_mask = cv2.dilate(border_mask, np.ones((5, 5), np.uint8), iterations=1)
    # blurred_mask = cv2.GaussianBlur(pred_final_resized, (5, 5), 0)  #(21,21)
    # pred_final_resized = np.where(border_mask > 0, blurred_mask, pred_final_resized)
    
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

# Start the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)