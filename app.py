import os
import sys
import time
import uuid
import json
import base64
import subprocess
import cv2
import numpy as np
from flask import Flask, render_template, request, session, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from DIP import detect_and_count_vehicles_accurate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, template_folder=BASE_DIR)
app.secret_key = "super_secret_offline_key"  # Required for developer login sessions

UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

# --- DEVELOPER CREDENTIALS ---
DEV_USERNAME = "admin"
DEV_PIN = "1234"
DEV_AUTHENTICATED = False  # Fallback for offline file:// requests where cookies are blocked


@app.after_request
def add_offline_cors_headers(response):
    """Allows index.html to talk to app.py even if opened directly via double-click (file://)."""
    origin = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Origin"] = origin if origin != "null" else "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/")
def serve_index():
    """Serves the offline website at http://127.0.0.1:5000"""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/temp_uploads/<path:filename>")
def serve_temp_uploads(filename):
    """Serves generated images from temp_uploads if referenced by path."""
    return send_from_directory(UPLOAD_DIR, filename)


# ==========================================
# 1. VEHICLE DETECTION ROUTES
# ==========================================
@app.route("/api/health", methods=["GET", "OPTIONS"])
def health_check():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    obb_exists = os.path.exists(os.path.join(BASE_DIR, "yolov8m-obb.pt"))
    std_exists = os.path.exists(os.path.join(BASE_DIR, "yolov8m.pt"))
    return jsonify({
        "status": "online",
        "backend": "DIP.py (OpenCV + Ultralytics YOLOv8)",
        "models_found": {
            "yolov8m-obb.pt": obb_exists,
            "yolov8m.pt": std_exists
        }
    })


@app.route("/api/detect", methods=["POST", "OPTIONS"])
def run_detection():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    start_time = time.time()
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded under field 'image'."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename received."}), 400

    # Read parameters from form
    conf_threshold = float(request.form.get("conf_threshold", 0.28))
    iou_threshold = float(request.form.get("iou_threshold", 0.40))
    scene_mode = request.form.get("scene_mode", "street")
    use_clahe = request.form.get("use_clahe", "true").lower() == "true"
    use_tiling = request.form.get("use_tiling", "true").lower() == "true"

    job_id = uuid.uuid4().hex[:8]
    input_path = os.path.join(UPLOAD_DIR, f"input_{job_id}.jpg")
    output_path = os.path.join(UPLOAD_DIR, f"accurate_output_{job_id}.jpg")

    try:
        file.save(input_path)

        result = detect_and_count_vehicles_accurate(
            image_path=input_path,
            output_path=output_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            scene_mode=scene_mode,
            use_clahe=use_clahe,
            use_tiling=use_tiling,
            show_window=False
        )

        with open(output_path, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode("utf-8")

        elapsed_ms = int((time.time() - start_time) * 1000)

        return jsonify({
            "success": True,
            "output_image": f"data:image/jpeg;base64,{encoded_img}",
            "total_vehicles": result["total_vehicles"],
            "counts": result["counts"],
            "raw_candidates": result["raw_candidates"],
            "tiles_processed": result["tiles_processed"],
            "image_size": result["image_size"],
            "detections": result["detections"],
            "processing_time_ms": elapsed_ms
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary files
        for p in (input_path, output_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ==========================================
# 2. STAR DETECTION ROUTE
# ==========================================
@app.route("/star-detection", methods=["GET", "POST", "OPTIONS"])
def star_detection():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Basic Digital Image Processing (DIP) for Star Detection
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({"error": "Could not read uploaded image file"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur and Thresholding to isolate bright stars
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # Find contours (stars)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        star_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 1:  # Filter out tiny noise
                star_count += 1
                (x, y), radius = cv2.minAreaRect(cnt)[0], 5
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        output_filename = "star_output_" + filename
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        cv2.imwrite(output_path, img)

        with open(output_path, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({
            "success": True,
            "message": "Star detection complete",
            "star_count": star_count,
            "output_image": f"data:image/jpeg;base64,{encoded_img}",
            "output_path": f"temp_uploads/{output_filename}"
        })

    # If GET request, render the page (or section)
    return render_template("index.html", active_tab="star")


# ==========================================
# 3. DEVELOPER AUTH & PYTHON MODEL ACCURACY
# ==========================================
@app.route("/dev-login", methods=["POST", "OPTIONS"])
def dev_login():
    global DEV_AUTHENTICATED
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(silent=True) or request.form
    username = data.get("username")
    pin = data.get("pin")

    if username == DEV_USERNAME and str(pin) == DEV_PIN:
        session["is_developer"] = True
        DEV_AUTHENTICATED = True
        return jsonify({"success": True, "message": "Developer access granted"})
    return jsonify({"success": False, "message": "Invalid Username or PIN"}), 401


@app.route("/dev-logout", methods=["POST", "OPTIONS"])
def dev_logout():
    global DEV_AUTHENTICATED
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    session.pop("is_developer", None)
    DEV_AUTHENTICATED = False
    return jsonify({"success": True})


@app.route("/upload-model-script", methods=["POST", "OPTIONS"])
def upload_model_script():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    # Ensure only logged-in developers can access this
    if not (session.get("is_developer") or DEV_AUTHENTICATED):
        return jsonify({"error": "Unauthorized. Developer login required."}), 403

    if "script" not in request.files:
        return jsonify({"error": "No Python file uploaded"}), 400

    file = request.files["script"]
    if not file.filename or not file.filename.endswith(".py"):
        return jsonify({"error": "Only .py files are allowed"}), 400

    filename = secure_filename(file.filename)
    script_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(script_path)

    try:
        # Run the uploaded Python script using the current Python interpreter
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=BASE_DIR
        )

        if result.returncode != 0:
            return jsonify({"error": "Script error", "details": result.stderr}), 400

        # Expecting the script to print JSON or text with model accuracies
        try:
            accuracy_data = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            # Fallback if the script prints plain text instead of JSON
            accuracy_data = {"raw_output": result.stdout}

        return jsonify({"success": True, "accuracies": accuracy_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 65)
    print("  OFFLINE VEHICLE & STAR DETECTION SERVER RUNNING")
    print("  Open in your browser: http://127.0.0.1:5000")
    print("  (Or open index.html directly from your folder)")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=False)