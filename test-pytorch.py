import cv2
import numpy as np
import torch
from unetV2 import UNetPlusInception  # Your PyTorch model file


def load_and_preprocess_image(image_path, target_size=(224, 224), device="cpu"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Could not read image at path: {image_path}")

    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img_tensor = torch.tensor(img, dtype=torch.float32).to(device)
    return img_tensor


def predict_segmentation(model_path, image_path, output_path="predicted_mask.png", device="cpu"):
    model = UNetPlusInception()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    img_tensor = load_and_preprocess_image(image_path, device=device)

    with torch.no_grad():
        prediction = model(img_tensor)
        prediction = prediction.cpu().numpy()

    # prediction shape: (1, 1, H, W)
    mask = (prediction[0, 0] > 0.5).astype(np.uint8) * 255

    cv2.imwrite(output_path, mask)
    print(f"Predicted mask saved to {output_path}")


if __name__ == "__main__":
    model_path = "/content/inception-unet/results-1/UNETV2_weights.pth"
    image_path = "/content/drive/MyDrive/Massachusetts/test/22828930_15.tiff"
    predict_segmentation(model_path, image_path)
