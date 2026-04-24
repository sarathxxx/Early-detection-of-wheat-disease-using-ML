"""
Wheat Disease Detection System
Model: Swin Transformer (swin_tiny_patch4_window7_224) via timm + PyTorch
9 Classes: Brown Rust, Crown and Root Rot, Fusarium Head Blight, Healthy,
           Leaf Rust, Loose Smut, Septoria, Stripe Rust, Yellow Rust

Run: python app.py
Place trained weights at: model/final_model.pth
"""
from model_setup import predict_image
import os
import os, json, uuid, io, random, warnings
from datetime import datetime
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, send_file)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import sqlite3

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "final_model.pth")

app = Flask(__name__)
app.secret_key = "wheat-swin-secret-key-2024"
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

# ─── 9 Disease classes (exact order from training: ImageFolder alphabetical sort) ──
# The notebook trained on: combined_wheat_disease_dataset with these classes:
CLASS_NAMES = [
    "Brown Rust",           # 0
    "Crown and Root Rot",   # 1
    "Fusarium Head Blight", # 2
    "Healthy",              # 3
    "Leaf Rust",            # 4
    "Loose Smut",           # 5
    "Septoria",             # 6
    "Stripe Rust",          # 7
    "Yellow Rust",          # 8
]
CLASS_CODES = [
    "brown_rust", "crown_root_rot", "fusarium", "healthy",
    "leaf_rust", "loose_smut", "septoria", "stripe_rust", "yellow_rust"
]
NUM_CLASSES = len(CLASS_NAMES)

# ─── Load Swin Transformer model ─────────────────────────────────────────────
MODEL = None
USING_REAL_MODEL = False

def load_model():
    global MODEL, USING_REAL_MODEL
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Model weights not found at {MODEL_PATH}")
        print("[INFO] Running in simulation mode. Place final_model.pth in model/ to enable real inference.")
        return None
    try:
        import torch
        import timm
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=NUM_CLASSES
        )
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        USING_REAL_MODEL = True
        print(f"[INFO] ✅ Swin Transformer model loaded successfully from {MODEL_PATH}")
        print(f"[INFO] Device: {device}  |  Classes: {NUM_CLASSES}")
        return model
    except ImportError as e:
        print(f"[WARN] torch/timm not installed: {e}. Running in simulation mode.")
        print("[HINT] pip install torch timm")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}. Running in simulation mode.")
        return None

# ─── Image preprocessing (matches training notebook exactly) ─────────────────
def preprocess_image(image_path):
    """
    Preprocessing identical to the training notebook:
      transforms.Resize((224, 224))
      transforms.ToTensor()
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    """
    try:
        import torch
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)   # shape: (1, 3, 224, 224)
        return tensor
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None

# ─── Prediction ───────────────────────────────────────────────────────────────
def predict_disease(image_path):
    """
    REAL model prediction (no simulation)
    Returns: (class_code, confidence, all_probs)
    """
    try:
        import torch
        import torch.nn.functional as F
        import timm
        import torchvision.transforms as transforms
        from PIL import Image

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=len(CLASS_NAMES)
        )

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        # Transform (same as training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        # Load image
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1).squeeze(0)

        idx = int(probs.argmax().item())
        confidence = float(probs[idx].item())

        return (
            CLASS_CODES[idx],
            round(confidence, 4),
            [round(p.item(), 4) for p in probs]
        )

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "healthy", 0.5, [0.1]*len(CLASS_NAMES)

# ─── Database ─────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "database", "wheat.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db(); c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL, role TEXT DEFAULT 'user',
        created TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS diseases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL, code TEXT UNIQUE NOT NULL,
        symptoms TEXT, treatment TEXT, prevention TEXT,
        severity TEXT DEFAULT 'Medium', description TEXT
    );
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, image_path TEXT,
        disease TEXT, confidence REAL, severity TEXT,
        all_probs TEXT,
        created TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, message TEXT, rating INTEGER,
        created TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS disease_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, image_path TEXT, description TEXT,
        status TEXT DEFAULT 'pending',
        created TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    # ── 9-class disease knowledge base ─────────────────────────────────────
    diseases = [
        ("Brown Rust", "brown_rust",
         "Small circular to oval orange-brown pustules on upper leaf surfaces surrounded by yellow halos. Pustules contain orange-red urediniospores. Pustule size typically 1–2 mm.",
         "Apply fungicides (Propiconazole 25% EC at 1 ml/L, Tebuconazole 25.9% EC). Spray at first sign of infection and repeat after 14 days. Remove and destroy heavily infected plant material.",
         "Grow resistant varieties. Apply preventive fungicides before infection. Practice crop rotation. Monitor fields during warm (15–25°C) and humid conditions (>95% RH).",
         "High",
         "Caused by Puccinia triticina. One of the most widespread wheat diseases globally, favored by warm humid weather."),

        ("Crown and Root Rot", "crown_root_rot",
         "Browning and rotting of crown and root tissues. Stunted plants with yellow-brown discoloration at base. 'White heads' in severe infections. Roots show brown lesions and decay.",
         "Seed treatment with fungicides (Carboxin + Thiram, Thiabendazole). Avoid waterlogged conditions. Remove and destroy infected plants. Apply Trichoderma-based biocontrol agents.",
         "Use disease-free certified seed. Ensure proper soil drainage. Practice long crop rotations (3+ years). Avoid excessive nitrogen. Plant at recommended seeding depths.",
         "High",
         "Caused by multiple fungi including Fusarium culmorum and Bipolaris sorokiniana. Soil-borne and seed-borne disease that weakens the root system."),

        ("Fusarium Head Blight", "fusarium",
         "Premature bleaching of one or more spikelets or entire spikes. Pink-orange fungal mycelium visible at base of glumes. Shriveled lightweight grains ('tombstone' kernels). Brown lesions on glumes.",
         "Apply fungicides at anthesis (Metconazole 60g/L, Prothioconazole + Tebuconazole). Spray must coincide with flowering. Avoid harvesting immature or infected grain.",
         "Crop rotation with non-host crops (soybean, sunflower). Plow under corn/wheat stubble. Use tolerant varieties. Avoid late planting. Monitor weather during heading — disease favored by warm wet conditions.",
         "Very High",
         "Caused by Fusarium graminearum. Produces deoxynivalenol (DON) mycotoxin — extremely dangerous to humans and animals. Major economic threat worldwide."),

        ("Healthy", "healthy",
         "Uniform deep-green leaves with no visible lesions, pustules, spots, or discoloration. Vigorous upright growth. No yellowing or wilting. Clean stems and ears.",
         "No treatment needed. Continue regular agronomic practices: balanced fertilization (NPK), timely irrigation, and preventive monitoring.",
         "Maintain good agricultural practices: soil testing, balanced fertilization, timely sowing, proper crop spacing, and regular field scouting.",
         "None",
         "Wheat plant exhibiting optimal health. Continue preventive monitoring to maintain disease-free status throughout the growing season."),

        ("Leaf Rust", "leaf_rust",
         "Small round to slightly elongated orange-brown uredia (pustules) scattered on leaf surface, primarily on upper side. Surrounded by yellow chlorotic halo. Leaves may yellow and senesce prematurely.",
         "Apply triazole fungicides (Propiconazole, Flutriafol) or strobilurin+triazole mixtures. Apply early when <5% leaf area infected. Repeat spray if re-infection occurs.",
         "Use rust-resistant cultivars. Apply preventive fungicide sprays during high-risk periods. Eliminate volunteer wheat plants (green bridge). Avoid excessive nitrogen.",
         "High",
         "Caused by Puccinia triticina (same as Brown Rust — may refer to the same pathogen in different contexts). Extremely common and economically significant globally."),

        ("Loose Smut", "loose_smut",
         "Entire grain replaced by a mass of dark olive-brown teliospores (smut). Infected plants emerge slightly earlier than healthy plants. Smut mass disperses at heading, leaving bare rachis.",
         "Seed treatment is the primary control: systemic fungicide seed dressing with Carboxin, Tebuconazole, or Triadimenol. Plant only certified smut-free seed.",
         "Use certified disease-free seed. Apply systemic fungicide seed treatment before planting. Avoid saving seed from infected fields. Quarantine infected areas.",
         "Medium",
         "Caused by Ustilago tritici. Seed-borne pathogen that infects flowers and remains dormant inside seed until the next season. Devastating if not controlled at planting."),

        ("Septoria", "septoria",
         "Irregular tan to brown blotches with yellowish margins on leaves. Dark brown pycnidia (fruiting bodies) visible as small dots within lesions. Disease progresses upward from lower leaves. Brown lesions may also appear on glumes.",
         "Apply fungicides (Azoxystrobin, Epoxiconazole, Prothioconazole) at early flag leaf stage. Timing is critical — apply before disease reaches upper canopy. Combine products to prevent resistance.",
         "Grow moderately resistant varieties. Plow or burn infected crop residues. Avoid early sowing in autumn. Practice crop rotation. Apply preventive fungicides during wet weather at tillering.",
         "Medium",
         "Caused by Zymoseptoria tritici (Septoria tritici blotch) and Parastagonospora nodorum (Septoria nodorum blotch). Very common in wet cool climates — major yield loss in Europe."),

        ("Stripe Rust", "stripe_rust",
         "Yellow-orange uredia arranged in distinctive yellow stripes along leaf veins. Stripes follow parallel vein pattern. Leaves turn yellow. Later, dark brown telia may form. Pustules contain yellow powder.",
         "Apply systemic fungicides (Triazoles: Propiconazole, Tebuconazole; or Strobilurins) at first sign. Act quickly — stripe rust spreads rapidly. Re-apply if infection pressure is high.",
         "Plant resistant or moderately resistant varieties — primary management tool. Apply fungicides preventively in high-risk seasons. Avoid excessive nitrogen. Monitor cool mountain regions where disease overwinters.",
         "High",
         "Caused by Puccinia striiformis f. sp. tritici. Favored by cool (7–12°C) moist conditions. Can be epidemic and cause >70% yield loss in susceptible varieties under ideal conditions."),

        ("Yellow Rust", "yellow_rust",
         "Yellow-orange pustules arranged in stripes running parallel to leaf veins. Affected leaves turn completely yellow. Pustules release yellow powdery spores. In late stages, black telia may form.",
         "Apply fungicides (Triazoles, Strobilurins) at earliest sign. Spray promptly — yellow rust can spread extremely fast. Use high volume spray for thorough coverage.",
         "Use certified resistant wheat varieties — most effective strategy. Apply preventive fungicides. Eliminate volunteer wheat. Crop rotation. Monitor high-altitude regions during cool humid conditions.",
         "High",
         "Caused by Puccinia striiformis. Often refers to the same disease as Stripe Rust in different naming conventions. Extremely damaging under cool, moist weather in spring."),
    ]

    for d in diseases:
        c.execute("""INSERT OR IGNORE INTO diseases
                     (name,code,symptoms,treatment,prevention,severity,description)
                     VALUES (?,?,?,?,?,?,?)""", d)

    admin_pw = generate_password_hash("admin123")
    c.execute("INSERT OR IGNORE INTO users (name,email,password,role) VALUES (?,?,?,?)",
              ("Admin","admin@wheat.com",admin_pw,"admin"))
    conn.commit(); conn.close()

# ─── Auth helpers ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*a, **kw):
        if "user_id" not in session: return redirect(url_for("login"))
        return f(*a, **kw)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*a, **kw):
        if "user_id" not in session or session.get("role") != "admin":
            flash("Admin access required.", "error"); return redirect(url_for("dashboard"))
        return f(*a, **kw)
    return decorated

def allowed_file(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

def get_disease_info(code):
    conn = get_db()
    row = conn.execute("SELECT * FROM diseases WHERE code=?", (code,)).fetchone()
    conn.close()
    return dict(row) if row else {}

# ─── PDF report ───────────────────────────────────────────────────────────────
def generate_pdf_report(pred_id, user):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.units import cm

    conn = get_db()
    row = conn.execute("SELECT * FROM predictions WHERE id=?", (pred_id,)).fetchone()
    conn.close()
    if not row: return None
    pred = dict(row)
    info = get_disease_info(pred["disease"])

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    GREEN = colors.HexColor("#2d6a4f"); LIGHT = colors.HexColor("#f0f7f4")

    ts  = ParagraphStyle("t", parent=styles["Heading1"], textColor=GREEN, fontSize=22, spaceAfter=4)
    ss  = ParagraphStyle("s", parent=styles["Normal"],  textColor=colors.HexColor("#52796f"), fontSize=10)
    hs  = ParagraphStyle("h", parent=styles["Heading2"], textColor=GREEN, fontSize=13, spaceBefore=14, spaceAfter=4)
    bs  = ParagraphStyle("b", parent=styles["Normal"],  fontSize=10, leading=15)
    fts = ParagraphStyle("f", parent=styles["Normal"],  textColor=colors.gray, fontSize=8)

    story = []
    story.append(Paragraph("🌾 Wheat Disease Detection Report", ts))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}  |  "
        f"Patient: {user['name']}  |  Model: Swin Transformer (9-class)", ss))
    story.append(HRFlowable(width="100%", thickness=2, color=GREEN, spaceAfter=10))

    model_note = "Swin Transformer AI" if USING_REAL_MODEL else "Simulation (place final_model.pth in model/ for real AI)"
    td = [
        ["Disease Detected",  info.get("name", pred["disease"])],
        ["Confidence Score",  f"{pred['confidence']*100:.1f}%"],
        ["Severity Level",    pred["severity"] or info.get("severity","N/A")],
        ["Analysis Model",    model_note],
        ["Analysis Date",     pred["created"][:16]],
        ["Report Generated",  datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]
    t = Table(td, colWidths=[5*cm, 11*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),LIGHT),("TEXTCOLOR",(0,0),(0,-1),GREEN),
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),10),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white,LIGHT]),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#cde0d4")),("PADDING",(0,0),(-1,-1),8)
    ]))
    story.append(t); story.append(Spacer(1,10))

    for label, key in [
        ("Disease Overview",        "description"),
        ("Observed Symptoms",       "symptoms"),
        ("Recommended Treatment",   "treatment"),
        ("Prevention Methods",      "prevention"),
    ]:
        if info.get(key):
            story.append(Paragraph(label, hs))
            story.append(Paragraph(info[key], bs))

    # Probability breakdown
    if pred.get("all_probs"):
        try:
            probs = json.loads(pred["all_probs"])
            story.append(Paragraph("AI Confidence Breakdown (All Classes)", hs))
            prob_data = [["Disease", "Confidence", "Bar"]]
            for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
                bar = "█" * int(prob * 30)
                prob_data.append([name, f"{prob*100:.1f}%", bar])
            pt = Table(prob_data, colWidths=[5*cm, 2.5*cm, 8.5*cm])
            pt.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),GREEN),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),
                ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#cde0d4")),("PADDING",(0,0),(-1,-1),5),
                ("FONTNAME",(2,1),(2,-1),"Courier"),
            ]))
            story.append(pt); story.append(Spacer(1,10))
        except Exception:
            pass

    story.append(Spacer(1,16)); story.append(HRFlowable(width="100%",thickness=1,color=GREEN))
    story.append(Spacer(1,6))
    story.append(Paragraph(
        "This report is generated by the Wheat Disease Detection System using a Swin Transformer "
        "deep learning model trained on 9364 wheat images across 9 disease categories. "
        "Consult a certified agricultural expert before making field management decisions.",
        fts))
    doc.build(story); buf.seek(0); return buf

# ─── Routes: Public ───────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html", using_real_model=USING_REAL_MODEL)

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        name  = request.form.get("name","").strip()
        email = request.form.get("email","").strip().lower()
        pw    = request.form.get("password","")
        if not name or not email or not pw:
            flash("All fields required.","error"); return render_template("auth/signup.html")
        conn = get_db()
        if conn.execute("SELECT id FROM users WHERE email=?",(email,)).fetchone():
            conn.close(); flash("Email already registered.","error"); return render_template("auth/signup.html")
        conn.execute("INSERT INTO users (name,email,password) VALUES (?,?,?)",
                     (name,email,generate_password_hash(pw)))
        conn.commit(); conn.close()
        flash("Account created! Please log in.","success"); return redirect(url_for("login"))
    return render_template("auth/signup.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        pw    = request.form.get("password","")
        conn = get_db(); user = conn.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone(); conn.close()
        if user and check_password_hash(user["password"],pw):
            session.update({"user_id":user["id"],"name":user["name"],"email":user["email"],"role":user["role"]})
            return redirect(url_for("admin_dashboard") if user["role"]=="admin" else url_for("dashboard"))
        flash("Invalid email or password.","error")
    return render_template("auth/login.html")

@app.route("/logout")
def logout(): session.clear(); return redirect(url_for("index"))

# ─── Routes: User ─────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    preds = conn.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY created DESC LIMIT 10",
        (session["user_id"],)).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?",(session["user_id"],)).fetchone()[0]
    conn.close()
    return render_template("user/dashboard.html", predictions=preds, total=total,
                           using_real_model=USING_REAL_MODEL)

@app.route("/predict", methods=["GET","POST"])
@login_required
def predict():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No image uploaded.","error"); return redirect(request.url)
        file = request.files["image"]
        if not file or not allowed_file(file.filename):
            flash("Invalid file type. Please upload PNG/JPG/JPEG/WEBP.","error"); return redirect(request.url)
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        code, conf, all_probs = predict_disease(save_path)
        info = get_disease_info(code)

        conn = get_db()
        cur = conn.execute(
            "INSERT INTO predictions (user_id,image_path,disease,confidence,severity,all_probs) VALUES (?,?,?,?,?,?)",
            (session["user_id"], filename, code, conf,
             info.get("severity","Medium"), json.dumps(all_probs)))
        pred_id = cur.lastrowid; conn.commit(); conn.close()
        return redirect(url_for("result", pred_id=pred_id))
    return render_template("user/predict.html", using_real_model=USING_REAL_MODEL)

@app.route("/result/<int:pred_id>")
@login_required
def result(pred_id):
    conn = get_db()
    pred = conn.execute(
        "SELECT * FROM predictions WHERE id=? AND user_id=?",
        (pred_id,session["user_id"])).fetchone()
    conn.close()
    if not pred: flash("Not found.","error"); return redirect(url_for("dashboard"))
    info = get_disease_info(pred["disease"])
    # Parse probability vector
    all_probs = []
    if pred["all_probs"]:
        try:
            all_probs = json.loads(pred["all_probs"])
        except Exception:
            pass
    # Build zipped list for template: [(class_name, code, prob_pct), ...]
    prob_breakdown = []
    for i, (name, code) in enumerate(zip(CLASS_NAMES, CLASS_CODES)):
        p = all_probs[i] if i < len(all_probs) else 0.0
        prob_breakdown.append({
            "name": name, "code": code,
            "prob": p, "pct": round(p * 100, 1),
            "is_main": code == pred["disease"]
        })
    # Sort by prob descending
    prob_breakdown.sort(key=lambda x: x["prob"], reverse=True)
    return render_template("user/result.html",
                           pred=dict(pred), info=info,
                           prob_breakdown=prob_breakdown,
                           class_names=CLASS_NAMES,
                           using_real_model=USING_REAL_MODEL)

@app.route("/history")
@login_required
def history():
    conn = get_db()
    preds = conn.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY created DESC",
        (session["user_id"],)).fetchall()
    conn.close()
    return render_template("user/history.html", predictions=preds)

@app.route("/report/<int:pred_id>")
@login_required
def download_report(pred_id):
    conn = get_db()
    pred = conn.execute(
        "SELECT * FROM predictions WHERE id=? AND user_id=?",
        (pred_id,session["user_id"])).fetchone()
    conn.close()
    if not pred: flash("Not found.","error"); return redirect(url_for("dashboard"))
    buf = generate_pdf_report(pred_id, {"name":session["name"],"email":session["email"]})
    if not buf: flash("PDF generation failed.","error"); return redirect(url_for("result",pred_id=pred_id))
    return send_file(buf, as_attachment=True,
                     download_name=f"wheat_report_{pred_id}.pdf",
                     mimetype="application/pdf")

@app.route("/feedback", methods=["GET","POST"])
@login_required
def feedback():
    if request.method == "POST":
        msg    = request.form.get("message","").strip()
        rating = request.form.get("rating",5)
        if not msg: flash("Please enter a message.","error"); return render_template("user/feedback.html")
        conn = get_db()
        conn.execute("INSERT INTO feedback (user_id,message,rating) VALUES (?,?,?)",
                     (session["user_id"],msg,rating))
        conn.commit(); conn.close()
        flash("Thank you for your feedback!","success"); return redirect(url_for("dashboard"))
    return render_template("user/feedback.html")

@app.route("/submit-disease", methods=["GET","POST"])
@login_required
def submit_disease():
    if request.method == "POST":
        desc = request.form.get("description","").strip()
        if "image" not in request.files or not desc:
            flash("Image and description required.","error")
            return render_template("user/submit_disease.html")
        file = request.files["image"]
        if file and allowed_file(file.filename):
            fn = f"req_{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file.save(os.path.join(app.config["UPLOAD_FOLDER"],fn))
            conn = get_db()
            conn.execute("INSERT INTO disease_requests (user_id,image_path,description) VALUES (?,?,?)",
                         (session["user_id"],fn,desc))
            conn.commit(); conn.close()
            flash("Disease submission sent for review!","success"); return redirect(url_for("dashboard"))
        flash("Invalid file type.","error")
    return render_template("user/submit_disease.html")

@app.route("/diseases")
@login_required
def diseases():
    conn = get_db()
    rows = conn.execute("SELECT * FROM diseases ORDER BY name").fetchall()
    conn.close()
    return render_template("user/diseases.html", diseases=rows)

# ─── Routes: Admin ────────────────────────────────────────────────────────────
@app.route("/admin")
@admin_required
def admin_dashboard():
    conn = get_db()
    stats = {
        "total_users":       conn.execute("SELECT COUNT(*) FROM users WHERE role='user'").fetchone()[0],
        "total_preds":       conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
        "total_diseases":    conn.execute("SELECT COUNT(*) FROM diseases").fetchone()[0],
        "pending_requests":  conn.execute("SELECT COUNT(*) FROM disease_requests WHERE status='pending'").fetchone()[0],
        "total_feedback":    conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0],
    }
    recent = conn.execute("""
        SELECT p.*,u.name as uname FROM predictions p
        LEFT JOIN users u ON p.user_id=u.id ORDER BY p.created DESC LIMIT 8
    """).fetchall()
    conn.close()
    return render_template("admin/dashboard.html", stats=stats, recent_preds=recent,
                           using_real_model=USING_REAL_MODEL, num_classes=NUM_CLASSES)

@app.route("/admin/users")
@admin_required
def admin_users():
    conn = get_db(); users = conn.execute("SELECT * FROM users ORDER BY created DESC").fetchall(); conn.close()
    return render_template("admin/users.html", users=users)

@app.route("/admin/users/delete/<int:uid>", methods=["POST"])
@admin_required
def admin_delete_user(uid):
    conn = get_db(); conn.execute("DELETE FROM users WHERE id=? AND role!='admin'",(uid,)); conn.commit(); conn.close()
    flash("User deleted.","success"); return redirect(url_for("admin_users"))

@app.route("/admin/predictions")
@admin_required
def admin_predictions():
    conn = get_db()
    preds = conn.execute("""
        SELECT p.*,u.name as uname FROM predictions p
        LEFT JOIN users u ON p.user_id=u.id ORDER BY p.created DESC
    """).fetchall(); conn.close()
    return render_template("admin/predictions.html", predictions=preds)

@app.route("/admin/diseases")
@admin_required
def admin_diseases():
    conn = get_db(); d = conn.execute("SELECT * FROM diseases ORDER BY name").fetchall(); conn.close()
    return render_template("admin/diseases.html", diseases=d)

@app.route("/admin/diseases/add", methods=["GET","POST"])
@admin_required
def admin_add_disease():
    if request.method == "POST":
        f = request.form
        conn = get_db()
        conn.execute("INSERT INTO diseases (name,code,symptoms,treatment,prevention,severity,description) VALUES (?,?,?,?,?,?,?)",
                     (f["name"],f["code"],f["symptoms"],f["treatment"],f["prevention"],f["severity"],f["description"]))
        conn.commit(); conn.close()
        flash("Disease added.","success"); return redirect(url_for("admin_diseases"))
    return render_template("admin/disease_form.html", disease=None)

@app.route("/admin/diseases/edit/<int:did>", methods=["GET","POST"])
@admin_required
def admin_edit_disease(did):
    conn = get_db(); disease = conn.execute("SELECT * FROM diseases WHERE id=?",(did,)).fetchone()
    if request.method == "POST":
        f = request.form
        conn.execute("UPDATE diseases SET name=?,symptoms=?,treatment=?,prevention=?,severity=?,description=? WHERE id=?",
                     (f["name"],f["symptoms"],f["treatment"],f["prevention"],f["severity"],f["description"],did))
        conn.commit(); conn.close()
        flash("Disease updated.","success"); return redirect(url_for("admin_diseases"))
    conn.close()
    return render_template("admin/disease_form.html", disease=dict(disease) if disease else None)

@app.route("/admin/diseases/delete/<int:did>", methods=["POST"])
@admin_required
def admin_delete_disease(did):
    conn = get_db(); conn.execute("DELETE FROM diseases WHERE id=?",(did,)); conn.commit(); conn.close()
    flash("Disease deleted.","success"); return redirect(url_for("admin_diseases"))

@app.route("/admin/requests")
@admin_required
def admin_requests():
    conn = get_db()
    reqs = conn.execute("""
        SELECT r.*,u.name as uname FROM disease_requests r
        LEFT JOIN users u ON r.user_id=u.id ORDER BY r.created DESC
    """).fetchall(); conn.close()
    return render_template("admin/requests.html", requests=reqs)

@app.route("/admin/requests/<int:rid>/<action>", methods=["POST"])
@admin_required
def admin_update_request(rid, action):
    if action in ("approve","reject"):
        conn = get_db(); conn.execute("UPDATE disease_requests SET status=? WHERE id=?",(action+"d",rid)); conn.commit(); conn.close()
        flash(f"Request {action}d.","success")
    return redirect(url_for("admin_requests"))

@app.route("/admin/feedback")
@admin_required
def admin_feedback():
    conn = get_db()
    fb = conn.execute("""
        SELECT f.*,u.name as uname FROM feedback f
        LEFT JOIN users u ON f.user_id=u.id ORDER BY f.created DESC
    """).fetchall(); conn.close()
    return render_template("admin/feedback.html", feedbacks=fb)

# ─── API ──────────────────────────────────────────────────────────────────────
@app.route("/api/stats")
@login_required
def api_stats():
    conn = get_db()
    rows = conn.execute(
        "SELECT disease,COUNT(*) as cnt FROM predictions WHERE user_id=? GROUP BY disease",
        (session["user_id"],)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/model-info")
def api_model_info():
    return jsonify({
        "model": "swin_tiny_patch4_window7_224",
        "framework": "PyTorch + timm",
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "using_real_model": USING_REAL_MODEL,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
    })

# ─── Startup ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    MODEL = load_model()
    print("\n" + "="*60)
    print("  🌾 Wheat Disease Detection System")
    print(f"  Model: Swin Transformer (swin_tiny_patch4_window7_224)")
    print(f"  Classes: {NUM_CLASSES} — {', '.join(CLASS_NAMES[:3])}...")
    print(f"  Real model: {'✅ YES' if USING_REAL_MODEL else '⚠️  NO (simulation mode)'}")
    if not USING_REAL_MODEL:
        print(f"  → Place final_model.pth in: {MODEL_PATH}")
    print(f"  URL: http://localhost:5000")
    print(f"  Admin: admin@wheat.com / admin123")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
