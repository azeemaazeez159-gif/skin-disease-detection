from flask import Flask, render_template, request, redirect, session, url_for, flash
import sqlite3
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime

app = Flask(__name__)
app.secret_key = "skin_scan_ultra_premium_key"

# -------- CONFIGURATION --------
MODEL_PATH = "model/cnn_model.keras"
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------- LOAD MODEL --------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"⚠️ Warning: Model file not found at {MODEL_PATH}. Please run train_model.py first.")

# Class mapping
classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
class_names = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

# -------- DATABASE HELPER --------
def get_db():
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# -------- AUTH DECORATOR --------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# -------- ROUTES --------

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        db.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")
    
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_pw = generate_password_hash(password)

        db = get_db()
        try:
            db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            db.commit()
            flash("Registration successful! You can now login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
        finally:
            db.close()

    return render_template("register.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=session.get("username"))

@app.route("/upload")
@login_required
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "image" not in request.files:
        flash("No file part", "warning")
        return redirect(url_for("upload"))

    file = request.files["image"]
    if file.filename == "":
        flash("No image selected for uploading", "warning")
        return redirect(url_for("upload"))

    if model is None:
        return "Model not loaded. Please ensure cnn_model.keras exists in the model folder."

    # Save uploaded image
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Image Preprocessing
    img = Image.open(file_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)[0]
    
    # Get top 3 predictions
    top_indices = preds.argsort()[-3:][::-1]
    results = []
    for i in top_indices:
        results.append({
            "class": class_names[classes[i]],
            "confidence": round(float(preds[i]) * 100, 2)
        })

    top_disease = results[0]["class"]
    top_confidence = results[0]["confidence"]

    # Uncertainty check (threshold < 50%)
    uncertain = top_confidence < 50

    # Save to database
    db = get_db()
    db.execute(
        "INSERT INTO predictions (user_id, image_path, disease, confidence) VALUES (?, ?, ?, ?)",
        (session["user_id"], filename, top_disease, top_confidence)
    )
    db.commit()
    db.close()

    return render_template(
        "result.html",
        results=results,
        image=filename,
        uncertain=uncertain
    )

@app.route("/history")
@login_required
def history():
    db = get_db()
    predictions = db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC", 
        (session["user_id"],)
    ).fetchall()
    db.close()
    return render_template("history.html", predictions=predictions)

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)