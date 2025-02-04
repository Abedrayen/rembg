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

MEAN = torch.tensor([0.25, 0.25, 0.25])
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
    mask = generate_mask(image)
    final_image = remove_background(image, mask)
    img_io = BytesIO()
    final_image.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', as_attachment=True, download_name="background_removed.png")

if __name__ == '__main__':
    app.run(debug=True)
