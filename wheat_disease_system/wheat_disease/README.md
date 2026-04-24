# 🌾 Wheat Disease Detection System

**Model: Swin Transformer (`swin_tiny_patch4_window7_224`) — PyTorch + timm**  
**9 Disease Classes | 9,364 Training Images**

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask werkzeug pillow numpy reportlab
pip install torch torchvision timm   # For real AI inference
```

### 2. Place Your Trained Model
Copy `final_model.pth` (exported from the Kaggle notebook) into the `model/` folder:
```
wheat_disease/
└── model/
    └── final_model.pth   ← place here
```

### 3. Run
```bash
python app.py
```
Visit **http://localhost:5000**

---

## 🧠 Model Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | Swin Transformer Tiny |
| timm model name | `swin_tiny_patch4_window7_224` |
| Input size | 224 × 224 RGB |
| Normalization | mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] |
| Number of classes | 9 |
| Training images | 9,364 |
| Framework | PyTorch + timm |

### 9 Disease Classes (in training order)
| Index | Class Name | Code |
|-------|-----------|------|
| 0 | Brown Rust | `brown_rust` |
| 1 | Crown and Root Rot | `crown_root_rot` |
| 2 | Fusarium Head Blight | `fusarium` |
| 3 | Healthy | `healthy` |
| 4 | Leaf Rust | `leaf_rust` |
| 5 | Loose Smut | `loose_smut` |
| 6 | Septoria | `septoria` |
| 7 | Stripe Rust | `stripe_rust` |
| 8 | Yellow Rust | `yellow_rust` |

---

## 🔑 Default Credentials

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@wheat.com | admin123 |

---

## 📁 Project Structure

```
wheat_disease/
├── app.py                    # Main Flask app (model integration + all routes)
├── requirements.txt
├── README.md
├── model/
│   └── final_model.pth       # ← Your trained Swin Transformer weights
├── database/
│   └── wheat.db              # Auto-created SQLite database
├── static/
│   ├── css/style.css
│   ├── css/landing.css
│   ├── js/main.js
│   └── uploads/              # Uploaded images stored here
└── templates/
    ├── base.html
    ├── index.html
    ├── auth/{login,signup}.html
    ├── user/{dashboard,predict,result,history,diseases,feedback,submit_disease}.html
    └── admin/{dashboard,users,predictions,diseases,disease_form,requests,feedback}.html
```

---

## 🔗 Exporting the Model from Kaggle

In the Kaggle notebook, the model is saved as:
```python
torch.save(model.state_dict(), "/content/final_model.pth")
```

Download `final_model.pth` from Kaggle and place it in `model/`.

---

## ⚙️ How the Inference Works (app.py)

```python
import torch, timm
import torchvision.transforms as transforms

# Load model
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=9)
model.load_state_dict(torch.load('model/final_model.pth', map_location='cpu'))
model.eval()

# Preprocess (matches training exactly)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Predict
img = Image.open(image_path).convert('RGB')
tensor = transform(img).unsqueeze(0)
with torch.no_grad():
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze()
class_idx = probs.argmax().item()
confidence = probs[class_idx].item()
```

---

## 🌐 API Endpoints

- `GET /api/model-info` — Returns model info, class list, whether real model is active
- `GET /api/stats` — User's prediction statistics

---

## 🔒 Security
- Password hashing via Werkzeug PBKDF2
- Session-based authentication with role-based access
- File type validation on all uploads
- Change `app.secret_key` before production use

---

## 📊 Features
- Upload wheat leaf image → instant diagnosis
- Full 9-class softmax probability breakdown
- Downloadable PDF report with treatment + prevention
- Prediction history with image thumbnails
- Disease library with symptom/treatment info
- User feedback system (1–5 stars)
- Disease submission for admin review
- Full admin panel (users, predictions, diseases, requests, feedback)
