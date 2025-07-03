# Traffic Sign


- EfficientDet-D1 for object detection and bounding box generation.
- A finetuned CNN classifier or a pretrained-only classifier for sign classification.
- Plug-and-play architecture – you only need to update a few paths to get started.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/TrafficSignDetection.git
cd TrafficSignDetection
````

### 2. Install Dependencies

Ensure you have Python 3.7+ and the required libraries installed. Run:

```bash
pip install -r requirements.txt
```

---

## Files Overview

| File Name                         | Purpose                                                    |
| --------------------------------- | ---------------------------------------------------------- |
| `videoclassification17classes.py` | Run detection and classification on videos                 |
| `imageclassification17classes.py` | Run detection and classification on images                 |
| `Class_and_detect.ipynb`          | For inference using only pretrained models (no finetuning) |
| `tl_retrained_classifier_17.h5`   | CNN model trained with transfer learning and fine-tuning   |
| `my_model.h5`                     | CNN model trained only on base pretrained weights          |
| `README.md`                       | Project documentation                                      |

---

## Change These Paths Before Running Scripts

Update the following paths in your script before running:

```python
EFFICIENTDET_PATH = r"C:\Users\Abinaya\Downloads\TrafficSignDetection-main\TrafficSignDetection-main\saved_models\efficientdet_d1_coco17_tpu-32\saved_model"
CLASSIFIER_PATH = r"C:\PROJECTS\TrafficSignIdentification\tl_retrained_classifier_17.h5"
VIDEO_PATH = r"C:\PROJECTS\TrafficSignIdentification\drivingdata\passenger_view.mp4"
OUTPUT_VIDEO_PATH = r"C:\PROJECTS\TrafficSignIdentification\output\classifiermodeltlandfinetunedmethod2\17classes\passengerview\passengerview.mp4"
```

For the model without transfer learning and finetuning, use:

```python
CLASSIFIER_PATH = r"path\to\my_model.h5"
```

---

## Behind the Scenes

### Detection Model: EfficientDet-D1

* Pretrained on the COCO dataset
* Outputs:

  * Bounding box coordinates
  * Class labels
  * Confidence scores
* These boxes are used to crop out the detected traffic signs.

### Classification Model: CNN

* Input: Cropped traffic signs
* Output: Predicted traffic sign label
* Two variants:

  1. Finetuned + Transfer Learned model (`tl_retrained_classifier_17.h5`)
  2. Only Pretrained model (`my_model.h5`)

---

## Run the Project

### For Video Classification (Transfer Learned and Finetuned):

```bash
python videoclassification17classes.py
```

### For Image Classification (Transfer Learned and Finetuned):

```bash
python imageclassification17classes.py
```

### For Pretrained-Only Model (Basic Detection and Classification):

Open and run:

```
Class_and_detect.ipynb
```

---

## Model Sources

* EfficientDet-D1:
  [https://github.com/ngonhi/TrafficSignDetection.git](https://github.com/ngonhi/TrafficSignDetection.git)
  saved\_models/efficientdet\_d1\_coco17\_tpu-32/saved\_model/

* Pretrained-only Classifier:
  [https://github.com/ItsCosmas/Traffic-Sign-Classification](https://github.com/ItsCosmas/Traffic-Sign-Classification)
  my\_model.h5

---

## Notes

* Bounding boxes are drawn using EfficientDet predictions.
* Cropped images from bounding boxes are passed into the classifier.
* Make sure to update paths correctly in the scripts before running them.
