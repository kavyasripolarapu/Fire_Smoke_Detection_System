# Fire & Smoke Detection System (Python)


---

## Overview
This project implements a simple **image-based fire and smoke detection system** using Python and TensorFlow.  
The model classifies images into three categories: **Fire**, **Smoke**, and **Normal**.

---

## Features
- 🔥 Detects Fire in images  
- ☁️ Detects Smoke in images  
- 🚫 Identifies Normal (no fire/smoke)  
- ✅ Uses CNN (Convolutional Neural Network)  
- 🖼️ Includes dataset verification and sample prediction visualization  

---

## Folder Structure
Fire_Smoke_Detection_System/
├── Fire_Smoke_Detection.py
├── fire_smoke_model.h5
├── fire_smoke_dataset/
│    ├── fire/
│    ├── smoke/
│    └── normal/
└── README.md

---

## How to Run

1. Open `Fire_Smoke_Detection.ipynb` in **Google Colab** or **VS Code Jupyter Notebook**.  
2. Make sure `fire_smoke_dataset/` is available in the notebook path.  
3. Run cells **in order**: data loading → preprocessing → model training → evaluation → sample prediction.  

---

## Model Performance

- Test Accuracy: **82.92%**  
- Observations: Most misclassifications occur between smoke and normal images.  

---

## Future Improvements

- Increase dataset size  
- Implement real-time video detection  
- Add web interface (Flask) for live upload & detection  
- Add data augmentation for improved generalization  

---
