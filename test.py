import tensorflow as tf
import numpy as np
import cv2  # For image processing
import unet, Inception, unetV2

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Could not read image at path: {image_path}")
    
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_segmentation(model_path, image_path, output_path="predicted_mask.png"):
    model = unetV2.get_unet_plus_inception()
    model.load_weights(model_path)

    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)

    mask = (prediction[0] > 0.5).astype(np.uint8) * 255
    mask = np.squeeze(mask, axis=-1)
    cv2.imwrite(output_path, mask)
    print(f"Predicted mask saved to {output_path}")

# Google colab path
model_path = '/content/inception-unet/results-1/UNETV2UNETV2.weights.h5' 
image_path = '/content/drive/MyDrive/Massachusetts/test/22828930_15.tiff' 
# test_image = np.zeros((224,224,3), dtype=np.uint8)

# cv2.imwrite(image_path, test_image)

predict_segmentation(model_path, image_path)
