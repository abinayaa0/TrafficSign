import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import time

# === Paths ===
EFFICIENTDET_PATH = r"C:\Users\Abinaya\Downloads\TrafficSignDetection-main\TrafficSignDetection-main\saved_models\efficientdet_d1_coco17_tpu-32\saved_model"
CLASSIFIER_PATH = r"C:\PROJECTS\TrafficSignIdentification\tl_retrained_classifier_17.h5"
INPUT_FOLDER = r"C:\PROJECTS\TrafficSignIdentification\drivingdata\sivaraman_images"  # make sure it's a folder with images
OUTPUT_FOLDER = r"C:\PROJECTS\TrafficSignIdentification\output\classifiermodeltlandfinetunedmethod2\17classes\Output3"

# === Constants ===
IMG_SIZE = 32
SCORE_THRESHOLD = 0.3
CLASSIFIER_THRESHOLD = 0.3

# === Load models ===
print(" Loading models...")
detect_fn = tf.saved_model.load(EFFICIENTDET_PATH)
classifier = load_model(CLASSIFIER_PATH)
print(" Models loaded successfully.")

# === 17-Class Names ===
classifier_classes = [
    "busstop",
    "giveway",
    "height-weight limit",
    "humpanddip",
    "menatwork",
    "no left turn",
    "no right turn",
    "noentry",
    "noovertaking",
    "noparking",
    "onewaytraffic",
    "parking",
    "roundabout",
    "speedlimit",
    "stop",
    "Tjunction",
    "zebracrossing"
]

# === Prepare output folder ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === List image files ===
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

start_time = time.time()
print("🚀 Starting image processing...")

with tqdm(total=len(image_files), desc=" Processing images") as pbar:
    for filename in image_files:
        img_path = os.path.join(INPUT_FOLDER, filename)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(image_rgb[tf.newaxis, ...], dtype=tf.uint8)

        detections = detect_fn(input_tensor)
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        
        for i in range(len(scores)):
            if scores[i] < SCORE_THRESHOLD:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            top, left = int(ymin * h), int(xmin * w)
            bottom, right = int(ymax * h), int(xmax * w)
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            normalized = gray.astype('float32') / 255.0
            normalized = np.expand_dims(normalized, axis=-1)
            pred = classifier.predict(np.expand_dims(normalized, axis=0), verbose=0)
            class_prob = np.max(pred)
            class_idx = np.argmax(pred)

            label = classifier_classes[class_idx] if class_prob >= CLASSIFIER_THRESHOLD else "Uncertain"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        out_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(out_path, frame)
        pbar.update(1)

# === Summary ===
total_time = time.time() - start_time
avg_time = total_time / len(image_files) if image_files else 0

print(f"\n All images processed and saved to: {OUTPUT_FOLDER}")
print(f" Total processing time: {total_time:.2f} seconds")
print(f" Average time per image: {avg_time:.3f} seconds")
