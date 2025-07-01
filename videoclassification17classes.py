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
VIDEO_PATH = r"C:\PROJECTS\TrafficSignIdentification\drivingdata\passenger_view.mp4"
OUTPUT_VIDEO_PATH = r"C:\PROJECTS\TrafficSignIdentification\output\classifiermodeltlandfinetunedmethod2\17classes\passengerview\passengerview.mp4"

# === Constants ===
IMG_SIZE = 32
NUM_CLASSES = 17
SCORE_THRESHOLD = 0.4
CLASSIFIER_THRESHOLD = 0.4

# === Load models ===
print("Loading models...")
detect_fn = tf.saved_model.load(EFFICIENTDET_PATH)
classifier = load_model(CLASSIFIER_PATH)
print("Models loaded successfully.")

# === Class names (17 total) ===
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

# === Initialize video capture ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

total_time = 0
processed_frames = 0

print("🚀 Starting video processing...")
with tqdm(total=frame_count, desc="Processing video") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(image_rgb[tf.newaxis, ...], dtype=tf.uint8)

        detections = detect_fn(input_tensor)
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()

        h, w, _ = frame.shape
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

        out.write(frame)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        processed_frames += 1
        pbar.update(1)

cap.release()
out.release()

avg_time = total_time / processed_frames if processed_frames > 0 else 0
print(f"\nOutput video saved at: {OUTPUT_VIDEO_PATH}")
print(f" Average processing time per frame: {avg_time:.2f} seconds")
print(f" Approx FPS during processing: {1/avg_time:.2f} fps")

#EfficientDet finetuning