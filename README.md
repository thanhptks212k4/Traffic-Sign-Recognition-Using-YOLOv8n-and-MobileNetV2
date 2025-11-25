# 🚦 Traffic Sign Recognition: YOLOv8n + MobileNetV2

[cite_start]This project implements an efficient and high-speed **Traffic Sign Recognition** system by combining two state-of-the-art deep learning models [cite: 5, 4][cite_start]: **YOLOv8n** (You Only Look Once, nano version) for Object Detection [cite: 5] [cite_start]and **MobileNetV2** for Feature Extraction and Classification[cite: 5].

[cite_start]The goal is to build a fast, accurate recognition pipeline that is optimized for deployment on resource-constrained devices[cite: 74, 103, 192].

---

## ✨ Project Highlights

* [cite_start]**Hybrid Architecture (Pipeline):** Combines the detection speed of **YOLOv8n** [cite: 66, 70] [cite_start]with the lightweight feature extraction capability of **MobileNetV2**[cite: 66, 71].
* [cite_start]**Flexible Classification:** MobileNetV2 is trained from scratch [cite: 157] [cite_start]to produce **Feature Vectors (Embeddings)** of 1280 dimensions [cite: 193, 204][cite_start], which are compared using **Cosine Similarity** to recognize the sign[cite: 67, 208]. [cite_start]This approach ensures stability and expandability[cite: 210].
* [cite_start]**High Accuracy:** Achieved an overall classification **Accuracy** of **97%** on the test set [cite: 183, 245] [cite_start]and a detection **mAP50** of approximately **0.95**[cite: 150, 243].
* [cite_start]**Real-time Speed:** The average processing speed is approximately **40–50 ms per image** on a GPU [cite: 250][cite_start], meeting the requirements for real-time applications[cite: 74, 257].
* [cite_start]**Intuitive Interface:** A user interface (UI) was built using **Gradio** to allow for easy interaction and testing of the model[cite: 212, 213].

---

## 🛠️ Technology Stack & Libraries

| Technology/Library | Purpose |
| :--- | :--- |
| **Python** | Primary programming language |
| **YOLOv8n (Ultralytics)** | [cite_start]Object Detection model for traffic signs[cite: 90]. |
| **MobileNetV2** | [cite_start]Feature Extraction and Classification model[cite: 102]. |
| **Gradio** | [cite_start]Building the web-based User Interface (UI)[cite: 212]. |
| **GTSRB** | [cite_start]Standard benchmark dataset for training (German Traffic Sign Recognition Benchmark)[cite: 83]. |

---

## 💡 Theoretical Foundation

### 1. Object Detection: YOLOv8n

[cite_start]YOLOv8n is a real-time object detection model known for its speed and high accuracy[cite: 92, 94]. [cite_start]In this project, YOLOv8n acts as the first component in the pipeline, responsible for precisely localizing the traffic signs[cite: 154].

* [cite_start]**Training Results:** After 50 epochs [cite: 147, 242][cite_start], the model achieved **Precision** $\approx 0.96$, **Recall** $\approx 0.93$, **mAP50** $\approx 0.95$, and **mAP50-95** $\approx 0.90$ on the validation set[cite: 150].

### 2. Feature Extraction: MobileNetV2

[cite_start]MobileNetV2 is a Convolutional Neural Network (CNN) [cite: 103] [cite_start]optimized for devices with limited resources[cite: 103].

* [cite_start]**Architecture:** Utilizes **Depthwise Separable Convolution** [cite: 104] [cite_start]and **Inverted Residual Blocks**[cite: 106].
* [cite_start]**Role:** Trained from scratch [cite: 157][cite_start], its backbone is used to extract the **1280-dimensional feature vector** [cite: 193, 204] [cite_start]from the output of the Global Average Pooling layer[cite: 204].

* [cite_start]**Training Results:** After 20 epochs[cite: 172], the model achieved:
    * [cite_start]Overall **Accuracy**: **97%**[cite: 183, 191].
    * [cite_start]Average **Precision, Recall, and F1-score**: $\approx 0.97$[cite: 183].

---

## ⚙️ System Workflow (Pipeline)

[cite_start]The integrated system operates through the following steps[cite: 197]:

1.  [cite_start]**Input:** A traffic image is fed into the system[cite: 199].
2.  [cite_start]**Detection (YOLOv8n):** YOLOv8n detects the sign and outputs a **Bounding Box** defining the ROI[cite: 199, 200].
3.  [cite_start]**Image Cropping (ROI):** The ROI is cropped from the original image[cite: 201].
4.  [cite_start]**Feature Extraction (MobileNetV2):** The cropped image is passed through the MobileNetV2 backbone (without the Softmax layer) [cite: 203, 204] [cite_start]to generate the **1280-dimensional Embedding Vector**[cite: 204, 205].
5.  [cite_start]**Recognition:** This embedding vector is compared against a database of reference feature vectors using **Cosine Similarity** to identify the closest sign class[cite: 208].
6.  [cite_start]**Output:** The final recognized label is displayed on the Gradio interface[cite: 223].

---

## ⏭️ Future Development Directions

1.  [cite_start]**Knowledge Distillation:** Apply this technique to transfer knowledge from a larger model (e.g., ResNet50) to MobileNetV2, aiming to increase accuracy without substantially increasing size[cite: 238].
2.  [cite_start]**Data Expansion:** Expand the training dataset to include more diverse real-world conditions (weather, varying lighting, and viewpoints)[cite: 239].
3.  [cite_start]**ADAS Integration:** Expand the system to recognize other traffic context components (vehicles, lanes, pedestrians) to build a more comprehensive **Advanced Driver-Assistance System (ADAS)**[cite: 239, 261].

---

## 📄 Report Information

This report serves as the Final Project for the Pattern Recognition course.

* **Student:** Phạm Tiến Thành – 2251262645
* **Supervisor:** Assoc. Prof. Dr. Nguyễn Quang Hoan
* [cite_start]**University:** Thuy Loi University (TLU) [cite: 1]
* [cite_start]**Date:** 2025 [cite: 7]
