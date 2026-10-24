===========================================================================
100% OFFLINE VEHICLE DETECTION WEBSITE + PYTHON BACKEND (DIP.py)
===========================================================================

WHY THIS VERSION RUNS COMPLETELY WITHOUT INTERNET:
1. Zero External CDNs: `index.html` contains 100% embedded CSS, inline SVG
   icons, and native JavaScript. It does NOT request Tailwind CDN, Google
   Fonts, or any external scripts.
2. Offline YOLO Mode: `DIP.py` sets `YOLO_OFFLINE=true` and
   `ULTRALYTICS_OFFLINE=true` so it loads your local `.pt` weight files
   directly from disk without attempting to connect to GitHub.
3. Built-in Local Server: `app.py` uses pure Flask (no `flask-cors` needed)
   and serves both the website and the `/api/detect` endpoint on your
   computer's localhost (`http://127.0.0.1:5000`).

---------------------------------------------------------------------------
HOW TO RUN OFFLINE:
---------------------------------------------------------------------------
1. Extract this ZIP file into the folder where your `yolov8m.pt` and
   `yolov8m-obb.pt` files are located:
     - index.html
     - app.py
     - DIP.py
     - run_offline_windows.bat
     - yolov8m.pt
     - yolov8m-obb.pt

2. Double-click `run_offline_windows.bat` (or run `python app.py` in your
   terminal).

3. Open `http://127.0.0.1:5000` in your browser (or double-click `index.html`
   directly).

4. Import any image:
   - The image immediately previews in the viewport.
   - A "Loading..." overlay with a dual-ring buffer spinner and scanning
     laser beam appears directly on top of the previewing image while
     uploading and while `DIP.py` processes the image.
   - Once complete, the annotated image (`accurate_output.jpg`) and vehicle
     counts are displayed and ready for 1-click download.
===========================================================================
