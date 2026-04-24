# 🌾 Wheat Disease Detection System using Swin Transformer

A full-stack AI-powered web application for **early detection of wheat diseases** using **Deep Learning (Swin Transformer)** and **Flask**. This project helps identify wheat leaf diseases from uploaded images and provides disease details, treatment suggestions, prevention methods, PDF reports, prediction history, and an admin management system.

---

## 🚀 Project Overview

Wheat is one of the most important crops worldwide, and early disease detection is critical for improving crop yield and preventing major agricultural losses.

This project uses a **Swin Transformer (Vision Transformer)** model trained on a wheat disease dataset to classify wheat leaf diseases into **9 different classes**.

The trained model is integrated into a Flask-based web application where users can upload wheat leaf images and instantly receive:

* Disease prediction
* Confidence score
* Severity level
* Symptoms explanation
* Treatment recommendations
* Prevention methods
* PDF downloadable report
* Prediction history tracking

---

## 🧠 Model Used

### Swin Transformer

We used:

**swin_tiny_patch4_window7_224**

from the **timm** library with **PyTorch**

### Why Swin Transformer?

* Better performance than traditional CNNs
* Uses self-attention mechanism
* Efficient image understanding
* Strong feature extraction
* High accuracy for plant disease detection

---

## 📌 Disease Classes

The model detects the following 9 wheat diseases:

1. Brown Rust
2. Crown and Root Rot
3. Fusarium Head Blight
4. Healthy
5. Leaf Rust
6. Loose Smut
7. Septoria
8. Stripe Rust
9. Yellow Rust

---

## 🛠 Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* timm
* torchvision

### Backend

* Flask

### Frontend

* HTML
* CSS
* JavaScript

### Database

* SQLite

### Visualization

* Matplotlib
* Seaborn

### Image Processing

* Pillow (PIL)
* NumPy

### Report Generation

* ReportLab

### Development Environment

* Google Colab (Model Training)
* VS Code (Project Development)

---

## ✨ Features

### User Features

* User Registration & Login
* Upload wheat leaf images
* Real-time disease prediction
* Confidence score visualization
* Full disease explanation
* Treatment recommendations
* Prevention methods
* PDF report generation
* Prediction history
* Feedback submission

### Admin Features

* Admin Dashboard
* User Management
* Prediction Monitoring
* Disease Knowledge Base Management
* Disease Request Review
* Feedback Management

---

## 📂 Project Structure

```bash
wheat_disease_system/
│
├── wheat_disease/
│   ├── app.py
│   ├── model_setup.py
│   ├── requirements.txt
│   │
│   ├── model/
│   │   ├── final_model.pth
│   │   └── classes.json
│   │
│   ├── templates/
│   │   ├── admin/
│   │   ├── auth/
│   │   ├── user/
│   │   └── base.html
│   │
│   ├── static/
│   │   └── uploads/
│   │
│   └── database/
│       └── wheat.db
│
└── README.md
```

---

## ⚙️ Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/wheat-disease-detection.git
cd wheat-disease-detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Add Trained Model

Place your trained model here:

```bash
model/final_model.pth
```

---

## ▶️ Run the Project

```bash
python app.py
```

Open browser:

```bash
http://127.0.0.1:5000
```

---

## 🔐 Default Admin Login

### Email

[admin@wheat.com](mailto:admin@wheat.com)

### Password

admin123

---

## 📈 Future Improvements

* Mobile App Integration
* Real-time Camera Detection
* IoT-based Smart Agriculture System
* Cloud Deployment
* Multi-language Support
* Improved Accuracy using Ensemble Models

---

## 🎯 Conclusion

This project combines **Artificial Intelligence + Agriculture + Full Stack Development** to provide a practical solution for early wheat disease detection.

It helps farmers and agricultural experts make faster decisions, reduce crop loss, and improve farming productivity.

This is not just a machine learning model — it is a complete deployable AI system.

---

## 👨‍💻 Author

Developed as a Final Year Machine Learning Project

**Project Title:**
Early Detection of Wheat Disease using Swin Transformer

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub
