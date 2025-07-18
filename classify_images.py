import cv2
import tensorflow as tf
import numpy as np
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Paths
MODEL_PATH = "Model/ewaste_model_alexnet.h5"
IMG_SIZE = (227, 227)  # Updated to match AlexNet input size
CLASS_NAMES = ['E-WASTE', 'NON-E-WASTE']

def preprocess_image(path, img_size):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img_resized = cv2.resize(img, img_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0), img

def classify_image(model_path, image_path, img_size):
    model = tf.keras.models.load_model(model_path)
    input_image, original_image = preprocess_image(image_path, img_size)
    prediction = model.predict(input_image, verbose=0)[0][0]

    if prediction > 0.5:
        label = CLASS_NAMES[1]
        confidence = prediction
        color = (0, 255, 0)
    else:
        label = CLASS_NAMES[0]
        confidence = 1 - prediction
        color = (0, 0, 255)

    print(f"\nFile: {os.path.basename(image_path)}")
    print(f"Predicted Class: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

    # Display the image with label
    display_image = cv2.resize(original_image, (512, 512))
    cv2.putText(display_image, f"{label} ({confidence * 100:.2f}%)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Prediction - AlexNet", display_image)
    print("Close the image window to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    for i in range(5):
        print(f"\n[{i+1}/5] Enter path to image:")
        path = input(">> ").strip().strip('"')

        if os.path.isfile(path):
            classify_image(MODEL_PATH, path, IMG_SIZE)
        else:
            print(f"[ERROR] File not found: {path}")

if __name__ == "__main__":
    main()
