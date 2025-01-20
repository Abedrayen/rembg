from flask import Flask, request, send_file, jsonify
import os
from PIL import Image
import torch
import torchvision.transforms as T
from isnet import ISNetDIS
import numpy as np
from io import BytesIO
import cv2

app = Flask(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ISNET_MODEL_PATH = "isnet-general-use.pth"
isnet = ISNetDIS(in_ch=3, out_ch=1)

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

isnet = load_model(model=isnet, model_path=ISNET_MODEL_PATH, device=DEVICE)

MEAN = torch.tensor([0.5, 0.5, 0.5])
STD = torch.tensor([0.5, 0.5, 0.5])
resize_shape = (2048, 2048)  # Larger resize shape for better detail capture
transforms = T.Compose([T.Resize(resize_shape), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

def normPRED(predicted_map):
    ma = np.max(predicted_map)
    mi = np.min(predicted_map)
    map_normalize = (predicted_map - mi) / (ma - mi + 1e-5)
    return map_normalize

def remove_background(image, mask):
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    
    mask = cv2.GaussianBlur(normPRED(mask), (15, 15), 0)  
    
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    foreground = image_np * mask_3ch
    background = np.zeros_like(image_np)
    final_image = np.where(mask_3ch > 0.5, foreground, background).astype(np.uint8)
    
    return Image.fromarray(final_image)

@app.route('/remove_background', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return jsonify({"error": "No image file in the request"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400
    
    metadata = image.info
    original_size = image.size
    original_extension = file.filename.split('.')[-1].lower()
    if original_extension == 'tif':
        original_extension = 'TIFF'
    elif original_extension == 'jpg':
        original_extension = 'JPEG'
    
    image_trans = transforms(image).to(DEVICE)
    with torch.no_grad():
        result = isnet(image_trans.unsqueeze(0))
    
    pred_normalize = normPRED(result[0][0].cpu().numpy())
    pred_normalize_squeezed = np.squeeze(pred_normalize)
    pred_final_resized = cv2.resize(pred_normalize_squeezed, original_size, interpolation=cv2.INTER_LINEAR)
    
    mask_uint8 = (pred_final_resized * 255).astype(np.uint8)
    edges = cv2.Canny(mask_uint8, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edges_dilated = (edges_dilated / 255.0).astype(np.float32)
    
    pred_final_resized = pred_final_resized.astype(np.float32)
    pred_final_resized = cv2.add(pred_final_resized, edges_dilated)
    pred_final_resized = np.clip(pred_final_resized, 0, 1)

    border_mask = cv2.Canny((pred_final_resized * 255).astype(np.uint8), 50, 150)
    border_mask = cv2.dilate(border_mask, np.ones((5, 5), np.uint8), iterations=1)
    blurred_mask = cv2.GaussianBlur(pred_final_resized, (21, 21), 0)
    pred_final_resized = np.where(border_mask > 0, blurred_mask, pred_final_resized)
    
    alpha = np.clip(pred_final_resized, 0, 1)
    image_np = np.array(image)
    foreground = (image_np * alpha[:, :, None]).astype(np.uint8)
    background = np.zeros_like(image_np)
    final_image = cv2.addWeighted(foreground, 1, background, 0, 0)
    
    img_io = BytesIO()
    output_image = Image.fromarray(final_image)
    output_image.save(img_io, format=original_extension, **metadata)
    img_io.seek(0)
    
    return send_file(img_io, mimetype=f'image/{original_extension.lower()}', as_attachment=True, download_name=f"background_removed.{original_extension.lower()}")

if __name__ == '__main__':
    app.run(debug=True)
