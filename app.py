from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    flash,
    redirect,
    url_for,
    session,
)
import pickle
import numpy as np
import pandas as pd
import json
from reportlab.platypus import KeepTogether, CondPageBreak
from reportlab.lib.units import inch
import os
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.lib.units import mm
from PIL import Image as PILImage  # Add this import
import tempfile  # Add this import
import atexit
from datetime import datetime, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from icrawler.builtin import BingImageCrawler

import random
from werkzeug.utils import secure_filename
import sklearn  # noqa: F401
import joblib
from flask import render_template_string
import threading
import time

import shutil
import atexit
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, String, Circle
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.colors import HexColor, Color
import io


active_sessions = {}


# Update your app configuration
app = Flask(__name__)
app.secret_key = "your-secret-key-change-this-to-something-secure"  # Change this to a secure secret key
app.permanent_session_lifetime = 3600

# Configuration
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"csv", "txt", "pkl"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/images/diet_images", exist_ok=True)
os.makedirs("diet_pdfs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Global variable to track crawling status
crawling_status = {}


def crawl_food_image(food_name, max_retries=3):
    """Crawl a single food image using Bing image crawler"""
    try:
        # Clean the food name for filename
        safe_name = "".join(
            c for c in food_name.lower().replace(" ", "_") if c.isalnum() or c in "_-"
        )
        output_dir = f"static/images/diet_images/{safe_name}"
        image_path = f"{output_dir}/000001.jpg"

        # Check if image already exists
        if os.path.exists(image_path):
            return f"/static/images/diet_images/{safe_name}/000001.jpg"

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set up Bing crawler with retries
        for attempt in range(max_retries):
            try:
                crawler = BingImageCrawler(
                    storage={"root_dir": output_dir},
                    log_level=30,  # WARNING level to reduce verbose output
                    downloader_threads=10,
                    parser_threads=10,
                    feeder_threads=10,
                )

                # Search for food images with more specific query
                search_query = f"{food_name} food dish recipe"
                print(f"Searching Bing for: {search_query}")

                crawler.crawl(keyword=search_query, max_num=1, file_idx_offset=0)

                # Check if image was downloaded
                if os.path.exists(image_path):
                    print(f"Successfully downloaded image for: {food_name}")
                    return f"/static/images/diet_images/{safe_name}/000001.jpg"
                else:
                    print(f"Attempt {attempt + 1}: No image found for {food_name}")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {food_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait longer before retry for Bing

        # If all attempts failed, return placeholder
        print(f"All attempts failed for {food_name}, using placeholder")
        return "/static/images/placeholder.jpg"

    except Exception as e:
        print(f"Error crawling image for {food_name}: {e}")
        return "/static/images/placeholder.jpg"


def crawl_diet_images_async(diet_charts):
    """Asynchronously crawl images for all food items in diet charts"""
    global crawling_status
    try:
        print("Starting asynchronous Bing image crawling...")
        food_items = set()

        # Collect all unique food items
        for chart in diet_charts:
            for meal_type in ["breakfast", "lunch", "snacks", "dinner"]:
                if meal_type in chart and chart[meal_type]:
                    for item in chart[meal_type]:
                        if "name" in item:
                            food_items.add(item["name"])

        print(f"Found {len(food_items)} unique food items to crawl from Bing")

        # Crawl images for each food item
        for i, food_name in enumerate(food_items):
            if food_name not in crawling_status:
                crawling_status[food_name] = "crawling"
                try:
                    print(f"Crawling image {i + 1}/{len(food_items)}: {food_name}")
                    image_path = crawl_food_image(food_name)
                    crawling_status[food_name] = image_path
                    print(f"Crawled image for: {food_name} -> {image_path}")

                    # Small delay between requests to be respectful to Bing
                    time.sleep(1)

                except Exception as e:
                    print(f"Failed to crawl image for {food_name}: {e}")
                    crawling_status[food_name] = "/static/images/placeholder.jpg"

        print("Bing image crawling completed")

    except Exception as e:
        print(f"Error in async Bing image crawling: {e}")


def crawl_food_image_batch(food_name, max_retries=2):
    """Optimized function to crawl a single food image with better error handling"""
    try:
        # Clean the food name for filename
        safe_name = "".join(
            c for c in food_name.lower().replace(" ", "_") if c.isalnum() or c in "_-"
        )
        output_dir = f"static/images/diet_images/{safe_name}"
        image_path = f"{output_dir}/000001.jpg"

        # Check if image already exists
        if os.path.exists(image_path):
            return f"/static/images/diet_images/{safe_name}/000001.jpg"

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set up Bing crawler with optimized settings for batch processing
        for attempt in range(max_retries):
            try:
                crawler = BingImageCrawler(
                    storage={"root_dir": output_dir},
                    log_level=40,  # ERROR level only to reduce output
                    downloader_threads=20,  # Increased threads
                    parser_threads=20,  # Increased threads
                    feeder_threads=20,  # Increased threads
                )

                # Search for food images with specific query
                search_query = f"{food_name} food dish"

                crawler.crawl(keyword=search_query, max_num=1, file_idx_offset=0)

                # Check if image was downloaded
                if os.path.exists(image_path):
                    print(f"✓ Downloaded image for: {food_name}")
                    return f"/static/images/diet_images/{safe_name}/000001.jpg"
                else:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)  # Shorter wait between retries

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {food_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)

        # If all attempts failed, return placeholder
        print(f"All attempts failed for {food_name}, using placeholder")
        return "/static/images/placeholder.jpg"

    except Exception as e:
        print(f"Error crawling image for {food_name}: {e}")
        return "/static/images/placeholder.jpg"


def crawl_diet_images_async_optimized(diet_charts):
    """Asynchronously crawl images for all food items using concurrent threading"""
    global crawling_status
    try:
        print("Starting optimized concurrent Bing image crawling...")
        food_items = set()

        # Collect all unique food items
        for chart in diet_charts:
            for meal_type in ["breakfast", "lunch", "snacks", "dinner"]:
                if meal_type in chart and chart[meal_type]:
                    for item in chart[meal_type]:
                        if "name" in item:
                            food_items.add(item["name"])

        food_list = list(food_items)
        print(f"Found {len(food_list)} unique food items to crawl")

        # Mark all as crawling initially
        for food_name in food_list:
            if food_name not in crawling_status:
                crawling_status[food_name] = "crawling"

        # Use ThreadPoolExecutor for concurrent downloads
        # Limit to 10 concurrent downloads to be respectful to Bing
        max_workers = min(10, len(food_list))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all crawling tasks
            future_to_food = {
                executor.submit(crawl_food_image_batch, food_name): food_name
                for food_name in food_list
            }

            # Process completed downloads
            completed = 0
            for future in as_completed(future_to_food):
                food_name = future_to_food[future]
                try:
                    image_path = future.result()
                    crawling_status[food_name] = image_path
                    completed += 1
                    print(f"✓ Completed ({completed}/{len(food_list)}): {food_name}")

                except Exception as e:
                    print(f"✗ Failed to crawl {food_name}: {e}")
                    crawling_status[food_name] = "/static/images/placeholder.jpg"
                    completed += 1

        print(f"✅ Concurrent image crawling completed! Downloaded {completed} images")

    except Exception as e:
        print(f"Error in optimized async image crawling: {e}")


def start_image_crawling_for_diet_optimized(diet_charts):
    """Start optimized concurrent image crawling when diet is generated"""
    print("Starting immediate concurrent image crawling for generated diet...")
    threading.Thread(
        target=crawl_diet_images_async_optimized, args=(diet_charts,), daemon=True
    ).start()


# Alternative approach: Batch crawling with icrawler for multiple keywords at once
def crawl_multiple_foods_single_session(food_list, batch_size=10):
    """Alternative: Use single crawler session for multiple foods"""
    global crawling_status
    try:
        print(f"Starting batch crawling for {len(food_list)} foods...")

        # Process foods in batches
        for i in range(0, len(food_list), batch_size):
            batch = food_list[i : i + batch_size]
            print(f"Processing batch {i // batch_size + 1}: {batch}")

            # Mark batch as crawling
            for food_name in batch:
                crawling_status[food_name] = "crawling"

            # Use single crawler instance for the batch
            base_dir = "static/images/diet_images"
            os.makedirs(base_dir, exist_ok=True)

            crawler = BingImageCrawler(
                storage={"root_dir": base_dir},
                log_level=40,
                downloader_threads=30,
                parser_threads=30,
                feeder_threads=30,
            )

            # Crawl each food in the batch
            for food_name in batch:
                try:
                    safe_name = "".join(
                        c
                        for c in food_name.lower().replace(" ", "_")
                        if c.isalnum() or c in "_-"
                    )

                    # Check if already exists
                    expected_path = f"{base_dir}/{safe_name}/000001.jpg"
                    if os.path.exists(expected_path):
                        crawling_status[food_name] = (
                            f"/static/images/diet_images/{safe_name}/000001.jpg"
                        )
                        continue

                    # Create specific subfolder for this food
                    food_dir = f"{base_dir}/{safe_name}"
                    os.makedirs(food_dir, exist_ok=True)

                    # Update crawler storage for this food
                    crawler.storage.root_dir = food_dir

                    search_query = f"{food_name} food dish recipe"
                    crawler.crawl(keyword=search_query, max_num=1)

                    # Check if downloaded successfully
                    if os.path.exists(f"{food_dir}/000001.jpg"):
                        crawling_status[food_name] = (
                            f"/static/images/diet_images/{safe_name}/000001.jpg"
                        )
                        print(f"✓ Batch downloaded: {food_name}")
                    else:
                        crawling_status[food_name] = "/static/images/placeholder.jpg"
                        print(f"✗ Batch failed: {food_name}")

                except Exception as e:
                    print(f"Error in batch crawling {food_name}: {e}")
                    crawling_status[food_name] = "/static/images/placeholder.jpg"

            # Small delay between batches
            if i + batch_size < len(food_list):
                time.sleep(1)

        print("✅ Batch crawling completed!")

    except Exception as e:
        print(f"Error in batch crawling: {e}")


def start_batch_crawling_for_diet(diet_charts):
    """Start batch crawling approach"""

    def extract_and_crawl():
        food_items = set()
        for chart in diet_charts:
            for meal_type in ["breakfast", "lunch", "snacks", "dinner"]:
                if meal_type in chart and chart[meal_type]:
                    for item in chart[meal_type]:
                        if "name" in item:
                            food_items.add(item["name"])

        crawl_multiple_foods_single_session(list(food_items), batch_size=10)

    threading.Thread(target=extract_and_crawl, daemon=True).start()


# Add new route to get image for food item
@app.route("/get-food-image/<food_name>")
def get_food_image(food_name):
    """Get image path for a food item"""
    global crawling_status
    try:
        if food_name in crawling_status:
            if crawling_status[food_name] == "crawling":
                return jsonify({"status": "loading", "image_path": None})
            else:
                return jsonify(
                    {"status": "ready", "image_path": crawling_status[food_name]}
                )
        else:
            # Start crawling this image immediately
            def crawl_single():
                crawling_status[food_name] = "crawling"
                image_path = crawl_food_image(food_name)
                crawling_status[food_name] = image_path

            threading.Thread(target=crawl_single).start()
            return jsonify({"status": "loading", "image_path": None})

    except Exception as e:
        print(f"Error getting food image: {e}")
        return jsonify(
            {"status": "error", "image_path": "/static/images/placeholder.jpg"}
        )


# Create placeholder image if it doesn't exist
def create_placeholder_image():
    """Create a placeholder image file"""
    placeholder_path = "static/images/placeholder.jpg"
    if not os.path.exists(placeholder_path):
        os.makedirs("static/images", exist_ok=True)
        # Create a simple placeholder (you can replace this with an actual image file)
        with open(placeholder_path.replace(".jpg", ".txt"), "w") as f:
            f.write("placeholder")


# Load models and data
def load_models():
    """Load all required models and encoders from your trained model"""
    try:
        model_files = {
            "model": "models/heart_disease_model_20250906_075310.pkl",
            "encoders": "models/label_encoders_20250906_075310.pkl",
            "features": "models/feature_info_20250906_075310.pkl",
            "metadata": "models/model_metadata_20250906_075310.pkl",
        }

        loaded_models = {}
        for name, path in model_files.items():
            if os.path.exists(path):
                try:
                    # Try joblib first, then pickle
                    if path.endswith(".pkl"):
                        try:
                            loaded_models[name] = joblib.load(path)
                        except:
                            with open(path, "rb") as f:
                                loaded_models[name] = pickle.load(f)
                    print(f"✓ Loaded {name} from {path}")
                except Exception as e:
                    print(f"✗ Error loading {name}: {e}")
            else:
                print(f"✗ Warning: {path} not found")

        return loaded_models
    except Exception as e:
        print(f"Error loading models: {e}")
        return {}


def load_diet_plan():
    """Load diet plan JSON from data directory"""
    try:
        with open("data/Final_diet_plan.json", "r") as f:
            diet_data = json.load(f)
            print("✓ Diet plan loaded successfully")
            return diet_data
    except FileNotFoundError:
        print("✗ Diet plan file not found: data/Final_diet_plan.json")
        return None
    except Exception as e:
        print(f"✗ Error loading diet plan: {e}")
        return None


# Global variables
models = load_models()
diet_plan = load_diet_plan()


def predict_patient_risk(model, patient_data):
    """
    Predict heart disease risk using the loaded model

    Parameters:
    model: trained machine learning model (already loaded)
    patient_data: list of patient features

    Returns:
    dict: prediction results
    """
    try:
        # Convert to numpy array and reshape for prediction
        patient_array = np.array(patient_data).reshape(1, -1)

        # Get prediction probability
        probabilities = model.predict_proba(patient_array)[0]

        # Handle binary classification (assuming class 1 is positive for heart disease)
        if len(probabilities) == 2:
            risk_prob = probabilities[1] * 100  # probability of class 1 (disease)
        else:
            risk_prob = max(probabilities) * 100

        # Get prediction
        prediction = model.predict(patient_array)[0]

        # Determine confidence level based on probability
        max_prob = max(probabilities) * 100
        if max_prob > 80:
            confidence = "High"
        elif max_prob > 65:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Determine risk level and color based on disease probability
        if risk_prob > 70:
            risk_level = "High Risk"
            risk_color = "danger"
            risk_category = "high_risk"
        elif risk_prob > 30:
            risk_level = "Moderate Risk"
            risk_color = "warning"
            risk_category = "moderate_risk"
        else:
            risk_level = "Low Risk"
            risk_color = "success"
            risk_category = "low_risk"

        return {
            "risk_probability": round(risk_prob, 2),
            "risk_level": risk_level,
            "risk_category": risk_category,
            "risk_color": risk_color,
            "prediction": int(prediction),
            "confidence_level": confidence,
            "all_probabilities": [round(p * 100, 2) for p in probabilities],
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None


@app.route("/")
def index():
    """Home page"""
    return render_template("index.html")


@app.route("/register-diet-session", methods=["POST"])
def register_diet_session():
    """Register a new diet chart viewing session"""
    try:
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "created": datetime.now(),
            "last_ping": datetime.now(),
            "images_folder": "static/images/diet_images",
        }

        # Start cleanup monitoring for this session
        threading.Thread(
            target=monitor_session, args=(session_id,), daemon=True
        ).start()

        return jsonify(
            {
                "status": "success",
                "session_id": session_id,
                "message": "Diet session registered",
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/ping-session", methods=["POST"])
def ping_session():
    """Keep session alive with periodic pings"""
    try:
        data = request.get_json()
        session_id = data.get("session_id")

        if session_id in active_sessions:
            active_sessions[session_id]["last_ping"] = datetime.now()
            return jsonify({"status": "success", "message": "Session pinged"})
        else:
            return jsonify({"status": "error", "message": "Session not found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/cleanup-images", methods=["POST"])
def cleanup_images():
    """Immediate cleanup when user explicitly leaves"""
    global crawling_status
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id")

        # If session_id provided, mark it for cleanup
        if session_id and session_id in active_sessions:
            active_sessions[session_id]["cleanup_requested"] = True

        # Always attempt cleanup
        images_dir = "static/images/diet_images"
        if os.path.exists(images_dir):

            def delete_folder():
                try:
                    time.sleep(1)  # Small delay
                    if os.path.exists(images_dir):
                        shutil.rmtree(images_dir)
                        print(f"✓ Deleted diet images folder: {images_dir}")

                        # Clear crawling status cache
                        crawling_status.clear()
                        print("✓ Cleared crawling status cache")

                        # Clear all active sessions
                        active_sessions.clear()

                except Exception as e:
                    print(f"Error deleting folder: {e}")

            threading.Thread(target=delete_folder, daemon=True).start()
            return jsonify(
                {"status": "cleanup_started", "message": "Image cleanup initiated"}
            )
        else:
            return jsonify(
                {"status": "no_folder", "message": "No images folder to cleanup"}
            )

    except Exception as e:
        print(f"Error in cleanup: {e}")
        return jsonify({"status": "error", "message": str(e)})


def monitor_session(session_id):
    """Monitor session and cleanup if inactive for too long"""
    global crawling_status
    try:
        while session_id in active_sessions:
            session_data = active_sessions[session_id]
            last_ping = session_data["last_ping"]

            # Check if session is inactive for more than 30 seconds
            if datetime.now() - last_ping > timedelta(seconds=600):
                print(f"Session {session_id} inactive for >600s, cleaning up...")

                # Cleanup images
                images_dir = session_data["images_folder"]
                if os.path.exists(images_dir):
                    try:
                        shutil.rmtree(images_dir)
                        print(f"✓ Auto-cleanup: Deleted {images_dir}")

                        crawling_status.clear()

                    except Exception as e:
                        print(f"Auto-cleanup error: {e}")

                # Remove session
                active_sessions.pop(session_id, None)
                break

            # Check if explicit cleanup was requested
            if session_data.get("cleanup_requested"):
                print(f"Session {session_id} cleanup requested, cleaning up...")

                images_dir = session_data["images_folder"]
                if os.path.exists(images_dir):
                    try:
                        shutil.rmtree(images_dir)
                        print(f"✓ Requested cleanup: Deleted {images_dir}")

                        crawling_status.clear()

                    except Exception as e:
                        print(f"Requested cleanup error: {e}")

                active_sessions.pop(session_id, None)
                break

            # Sleep for 5 seconds before next check
            time.sleep(5)

    except Exception as e:
        print(f"Session monitoring error: {e}")
        active_sessions.pop(session_id, None)


# Cleanup old sessions periodically
def cleanup_old_sessions():
    """Clean up sessions older than 5 minutes"""
    try:
        current_time = datetime.now()
        sessions_to_remove = []

        for session_id, data in active_sessions.items():
            if current_time - data["created"] > timedelta(minutes=5):
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            print(f"Removing old session: {session_id}")
            active_sessions.pop(session_id, None)

    except Exception as e:
        print(f"Old session cleanup error: {e}")


# Run cleanup every 2 minutes
def schedule_cleanup():
    while True:
        time.sleep(120)  # 2 minutes
        cleanup_old_sessions()


# Start the background cleanup scheduler
threading.Thread(target=schedule_cleanup, daemon=True).start()


def cleanup_on_exit():
    """Clean up images folder when server shuts down"""
    global crawling_status
    try:
        images_dir = "static/images/diet_images"
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
            print("✓ Server shutdown: Cleaned up diet images folder")
        active_sessions.clear()
        crawling_status.clear()
    except Exception as e:
        print(f"Shutdown cleanup error: {e}")


atexit.register(cleanup_on_exit)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Heart disease prediction page with 20 features"""
    if request.method == "GET":
        # Check if there's an existing prediction in session
        if "prediction_result" in session:
            return render_template(
                "predict.html", existing_prediction=session["prediction_result"]
            )
        else:
            return render_template("predict.html")

    try:
        # Get all 20 features from form data
        patient_data = [
            float(request.form["age"]),  # 1. Age
            int(request.form["gender"]),  # 2. Gender
            float(request.form["blood_pressure"]),  # 3. Blood Pressure
            float(request.form["cholesterol"]),  # 4. Cholesterol Level
            int(request.form["exercise"]),  # 5. Exercise Habits
            int(request.form["smoking"]),  # 6. Smoking
            int(request.form["family_heart_disease"]),  # 7. Family Heart Disease
            int(request.form["diabetes"]),  # 8. Diabetes
            float(request.form["bmi"]),  # 9. BMI
            int(request.form["high_bp_history"]),  # 10. High BP History
            int(request.form["low_hdl"]),  # 11. Low HDL Cholesterol
            int(request.form["high_ldl"]),  # 12. High LDL Cholesterol
            int(request.form["alcohol"]),  # 13. Alcohol Consumption
            int(request.form["stress_level"]),  # 14. Stress Level
            float(request.form["sleep_hours"]),  # 15. Sleep Hours
            int(request.form["sugar_consumption"]),  # 16. Sugar Consumption
            float(request.form["triglycerides"]),  # 17. Triglyceride Level
            float(request.form["fasting_blood_sugar"]),  # 18. Fasting Blood Sugar
            float(request.form["crp_level"]),  # 19. CRP Level
            float(request.form["homocysteine"]),  # 20. Homocysteine Level
        ]

        # Create user_data_dict for display purposes
        user_data_dict = {
            "age": float(request.form["age"]),
            "gender": "Male" if int(request.form["gender"]) == 1 else "Female",
            "blood_pressure": float(request.form["blood_pressure"]),
            "cholesterol": float(request.form["cholesterol"]),
            "exercise": ["Sedentary", "Light", "Moderate", "Heavy"][
                int(request.form["exercise"])
            ],
            "smoking": "Yes" if int(request.form["smoking"]) == 1 else "No",
            "family_heart_disease": "Yes"
            if int(request.form["family_heart_disease"]) == 1
            else "No",
            "diabetes": "Yes" if int(request.form["diabetes"]) == 1 else "No",
            "bmi": float(request.form["bmi"]),
            "high_bp_history": "Yes"
            if int(request.form["high_bp_history"]) == 1
            else "No",
            "low_hdl": "Yes" if int(request.form["low_hdl"]) == 1 else "No",
            "high_ldl": "Yes" if int(request.form["high_ldl"]) == 1 else "No",
            "alcohol": ["None", "Light", "Moderate", "Heavy"][
                int(request.form["alcohol"])
            ],
            "stress_level": int(request.form["stress_level"]),
            "sleep_hours": float(request.form["sleep_hours"]),
            "sugar_consumption": ["Low", "Moderate", "High"][
                int(request.form["sugar_consumption"])
            ],
            "triglycerides": float(request.form["triglycerides"]),
            "fasting_blood_sugar": float(request.form["fasting_blood_sugar"]),
            "crp_level": float(request.form["crp_level"]),
            "homocysteine": float(request.form["homocysteine"]),
        }

        print(f"Patient data received: {len(patient_data)} features")

        # Make prediction using the loaded model
        if models and models.get("model"):
            prediction_result = predict_patient_risk(models["model"], patient_data)

            if prediction_result:
                # Save outcome for diet generation
                outcome = {
                    "risk_probability": prediction_result["risk_probability"],
                    "risk_level": prediction_result["risk_level"],
                    "risk_category": prediction_result["risk_category"],
                    "risk_color": prediction_result["risk_color"],
                    "prediction": prediction_result["prediction"],
                    "confidence_level": prediction_result["confidence_level"],
                    "user_data": user_data_dict,
                    "patient_features": patient_data,
                    "timestamp": datetime.now().isoformat(),
                    "feature_count": len(patient_data),
                }

                # Save to session (for persistence across page navigation)
                session["prediction_result"] = outcome
                session.permanent = True

                # Save using pickle (for diet generation)
                try:
                    with open("static/diet_images/outcome.pkl", "wb") as f:
                        pickle.dump(outcome, f)
                    print("✓ Outcome saved successfully")
                except Exception as e:
                    print(f"✗ Error saving outcome: {e}")

                return render_template(
                    "result.html", prediction=outcome, user_data=user_data_dict
                )
            else:
                flash("Error in prediction calculation. Please try again.", "error")
                return redirect(url_for("predict"))
        else:
            flash("Model not loaded properly. Please check model files.", "error")
            return redirect(url_for("predict"))

    except KeyError as e:
        missing_field = str(e).replace("'", "")
        flash(
            f"Missing required field: {missing_field}. Please fill all fields.", "error"
        )
        print(f"Missing form field: {e}")
        return redirect(url_for("predict"))
    except ValueError as e:
        flash(f"Invalid input: {str(e)}", "error")
        print(f"Value error: {e}")
        return redirect(url_for("predict"))
    except Exception as e:
        print(f"Prediction error: {e}")
        print(f"Available form fields: {list(request.form.keys())}")
        flash(f"Error in prediction: {str(e)}", "error")
        return redirect(url_for("predict"))


@app.route("/diet-plan")
def diet_plan_page():
    """Diet plan generation page"""
    try:
        # First check session for prediction result
        if "prediction_result" in session:
            outcome = session["prediction_result"]
            print(f"Loaded outcome from session: {outcome['risk_category']}")
        else:
            # Fallback to pickle file
            try:
                with open("static/diet_images/outcome.pkl", "rb") as f:
                    outcome = pickle.load(f)
                print(f"Loaded outcome from pickle: {outcome['risk_category']}")
            except FileNotFoundError:
                flash("Please complete heart disease prediction first.", "warning")
                return redirect(url_for("predict"))

        return render_template("diet_plan.html", outcome=outcome)
    except Exception as e:
        print(f"Error loading outcome: {e}")
        flash(f"Error loading prediction results: {str(e)}", "error")
        return redirect(url_for("predict"))


@app.route("/generate-diet", methods=["POST"])
def generate_diet():
    """Generate personalized diet charts based on risk assessment"""
    try:
        # Get user preferences
        is_vegetarian = request.form.get("diet_type") == "vegetarian"
        num_charts = int(request.form.get("num_charts", 3))

        # Validate number of charts
        if num_charts < 1 or num_charts > 10:
            num_charts = 3

        # Load outcome from session first, then pickle as fallback
        outcome = None
        if "prediction_result" in session:
            outcome = session["prediction_result"]
        else:
            with open("static/diet_images/outcome.pkl", "rb") as f:
                outcome = pickle.load(f)

        if not diet_plan:
            flash(
                "Diet plan data not available. Please check data/Final_diet_plan.json file.",
                "error",
            )
            return redirect(url_for("diet_plan_page"))

        # Get diet recommendation based on risk category
        risk_category = outcome["risk_category"]
        diet_type = "vegetarian" if is_vegetarian else "non_vegetarian"

        print(f"Generating diet for: {diet_type}, {risk_category}")

        try:
            # Access diet data from JSON structure
            diet_data = diet_plan["diet_plan"][diet_type][risk_category]
            print(f"Found diet data with meals: {list(diet_data.keys())}")
        except KeyError as e:
            print(f"KeyError accessing diet data: {e}")
            available_categories = list(
                diet_plan.get("diet_plan", {}).get(diet_type, {}).keys()
            )
            flash(
                f"No diet plan found for {risk_category}. Available categories: {available_categories}",
                "error",
            )
            return redirect(url_for("diet_plan_page"))

        # Generate random diet charts
        diet_charts = create_random_diet_charts(diet_data, num_charts)

        if not diet_charts:
            flash("Could not generate diet charts. Please check diet data.", "error")
            return redirect(url_for("diet_plan_page"))

        # Find recommended chart (one closest to average calories)
        if len(diet_charts) > 1:
            total_calories = [chart.get("total_calories", 0) for chart in diet_charts]
            avg_calories = sum(total_calories) / len(total_calories)
            closest_chart = min(
                diet_charts,
                key=lambda x: abs(x.get("total_calories", 0) - avg_calories),
            )
            recommended_chart_num = closest_chart["chart_number"]
        else:
            recommended_chart_num = diet_charts[0]["chart_number"]

        # Mark recommended chart
        for chart in diet_charts:
            chart["is_recommended"] = chart["chart_number"] == recommended_chart_num

        print(
            f"Generated {len(diet_charts)} diet charts, recommended: Chart {recommended_chart_num}"
        )

        # Store diet charts and preferences in session for PDF generation
        session["diet_charts"] = diet_charts
        session["is_vegetarian"] = is_vegetarian
        session.permanent = True

        # Start crawling images IMMEDIATELY when diet is generated
        start_image_crawling_for_diet_optimized(diet_charts)

        return render_template(
            "diet_charts.html",
            diet_charts=diet_charts,
            outcome=outcome,
            is_vegetarian=is_vegetarian,
        )

    except FileNotFoundError:
        flash("Please complete heart disease prediction first.", "warning")
        return redirect(url_for("predict"))
    except Exception as e:
        print(f"Error in generate_diet: {e}")
        flash(f"Error generating diet plan: {str(e)}", "error")
        return redirect(url_for("diet_plan_page"))


@app.route("/new-prediction")
def new_prediction():
    """Clear session and start new prediction"""
    session.pop("prediction_result", None)
    flash("Starting new prediction. Please fill in your information.", "info")
    return redirect(url_for("predict"))


def create_random_diet_charts(diet_data, num_charts=3):
    """Create random diet charts from available diet data"""
    if not diet_data:
        print("No diet data provided")
        return []

    # Define meal configuration (items per meal type)
    meal_config = {"breakfast": 2, "lunch": 3, "snacks": 2, "dinner": 3}

    diet_charts = []

    print(f"Creating {num_charts} diet charts from meals: {list(diet_data.keys())}")

    for chart_num in range(1, num_charts + 1):
        chart = {"chart_number": chart_num, "total_calories": 0}

        for meal_type, num_items in meal_config.items():
            if meal_type in diet_data and diet_data[meal_type]:
                available_items = diet_data[meal_type]

                # Ensure we don't try to sample more items than available
                items_to_select = min(num_items, len(available_items))

                if items_to_select > 0:
                    # Randomly select items for this meal
                    selected_items = random.sample(available_items, items_to_select)
                    chart[meal_type] = selected_items

                    # Calculate calories for this meal
                    meal_calories = sum(
                        item.get("calories", 0) for item in selected_items
                    )
                    chart["total_calories"] += meal_calories
                else:
                    chart[meal_type] = []
            else:
                chart[meal_type] = []
                print(f"Warning: No {meal_type} data available")

        diet_charts.append(chart)

    return diet_charts


@app.route("/download-pdf")
def download_pdf():
    """Generate and download professional PDF without images"""
    try:
        # Load outcome from session first, then pickle as fallback
        outcome = None
        if "prediction_result" in session:
            outcome = session["prediction_result"]
        else:
            try:
                with open("static/diet_images/outcome.pkl", "rb") as f:
                    outcome = pickle.load(f)
            except FileNotFoundError:
                flash("Please complete heart disease prediction first.", "warning")
                return redirect(url_for("predict"))

        # Get diet charts from session
        diet_charts = session.get("diet_charts")
        if not diet_charts:
            flash("Please generate diet charts first.", "warning")
            return redirect(url_for("diet_plan_page"))

        # Add vegetarian preference to outcome for PDF generation
        outcome["is_vegetarian"] = session.get("is_vegetarian", False)

        # Generate professional PDF
        buffer = io.BytesIO()
        generate_diet_pdf_professional(buffer, outcome, diet_charts)
        buffer.seek(0)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CarePulse_Diet_Plan_{timestamp}.pdf"

        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf",
        )

    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback

        traceback.print_exc()
        flash(f"Error generating PDF: {str(e)}", "error")
        return redirect(url_for("diet_plan_page"))


def generate_diet_pdf_professional(buffer, outcome, diet_charts):
    """Generate a clean, professional PDF with proper formatting and spacing"""
    from reportlab.platypus import KeepTogether, CondPageBreak
    from reportlab.lib.units import inch
    
    # Create PDF document with better margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    # Define professional colors
    primary_color = HexColor("#1f2937")  # Dark gray
    accent_color = HexColor("#3b82f6")  # Blue
    success_color = HexColor("#059669")  # Green
    warning_color = HexColor("#d97706")  # Orange
    danger_color = HexColor("#dc2626")  # Red
    light_gray = HexColor("#f9fafb")
    border_color = HexColor("#e5e7eb")

    risk_color_map = {
        "success": success_color,
        "warning": warning_color,
        "danger": danger_color,
        "primary": accent_color,
    }

    # Professional styles with better spacing
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ProfessionalTitle",
        parent=styles["Heading1"],
        fontSize=28,
        spaceAfter=30,
        spaceBefore=20,
        alignment=TA_CENTER,
        textColor=primary_color,
        fontName="Helvetica-Bold",
        leading=34,
    )

    subtitle_style = ParagraphStyle(
        "ProfessionalSubtitle",
        parent=styles["Heading2"],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=10,
        alignment=TA_CENTER,
        textColor=accent_color,
        fontName="Helvetica",
        leading=22,
    )

    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=16,
        spaceAfter=15,
        spaceBefore=25,
        textColor=primary_color,
        fontName="Helvetica-Bold",
        borderWidth=1,
        borderColor=accent_color,
        borderPadding=12,
        backColor=light_gray,
        leading=20,
        keepWithNext=True,  # Prevent orphaned headers
    )

    normal_style = ParagraphStyle(
        "ProfessionalNormal",
        parent=styles["Normal"],
        fontSize=11,
        spaceAfter=8,
        spaceBefore=4,
        fontName="Helvetica",
        leading=16,
        leftIndent=0,
        rightIndent=0,
    )

    meal_header_style = ParagraphStyle(
        "MealHeader",
        parent=styles["Heading3"],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=accent_color,
        fontName="Helvetica-Bold",
        leading=18,
        keepWithNext=True,
    )

    info_style = ParagraphStyle(
        "InfoStyle",
        parent=normal_style,
        fontSize=10,
        textColor=HexColor("#6b7280"),
        alignment=TA_CENTER,
        spaceAfter=12,
        spaceBefore=8,
    )

    story = []

    # ===== HEADER SECTION =====
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("CarePulse", title_style))
    story.append(Paragraph("Personalized Diet Plan Report", subtitle_style))
    
    generation_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"Generated on: {generation_date}", info_style))
    story.append(Spacer(1, 0.4 * inch))

    # ===== PATIENT INFORMATION SECTION =====
    story.append(Paragraph("Patient Information", section_header_style))

    patient_data = [
        ["Parameter", "Value"],
        ["Name", "Patient"],
        ["Age", f"{outcome['user_data']['age']} years"],
        ["Gender", outcome["user_data"]["gender"]],
        ["BMI", f"{outcome['user_data']['bmi']}"],
        [
            "Diet Type",
            "Vegetarian" if outcome.get("is_vegetarian", False) else "Non-Vegetarian",
        ],
    ]

    patient_table = Table(patient_data, colWidths=[3.2 * inch, 2.8 * inch])
    patient_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), accent_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("BACKGROUND", (0, 1), (-1, -1), light_gray),
            ("GRID", (0, 0), (-1, -1), 1, border_color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])
    )

    story.append(KeepTogether([patient_table]))
    story.append(Spacer(1, 0.3 * inch))

    # ===== RISK ASSESSMENT SECTION =====
    story.append(Paragraph("Risk Assessment", section_header_style))

    risk_data = [
        ["Assessment Parameter", "Result"],
        ["Heart Disease Risk Level", outcome["risk_level"]],
        ["Risk Probability", f"{outcome['risk_probability']}%"],
        ["Confidence Level", outcome["confidence_level"]],
        ["Risk Category", outcome["risk_category"].replace("_", " ").title()],
    ]

    risk_table = Table(risk_data, colWidths=[3.2 * inch, 2.8 * inch])
    risk_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), accent_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("BACKGROUND", (0, 1), (-1, -1), light_gray),
            ("GRID", (0, 0), (-1, -1), 1, border_color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            # Highlight risk level row
            ("BACKGROUND", (1, 1), (1, 1), risk_color_map.get(outcome["risk_color"], accent_color)),
            ("TEXTCOLOR", (1, 1), (1, 1), colors.white),
            ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
        ])
    )

    story.append(KeepTogether([risk_table]))
    story.append(Spacer(1, 0.3 * inch))

    # ===== DIET PLAN OVERVIEW =====
    story.append(Paragraph("Diet Plan Overview", section_header_style))

    total_charts = len(diet_charts)
    calories_list = [chart["total_calories"] for chart in diet_charts]
    avg_calories = sum(calories_list) / len(calories_list) if calories_list else 0
    min_calories = min(calories_list) if calories_list else 0
    max_calories = max(calories_list) if calories_list else 0
    recommended_chart = next(
        (i + 1 for i, chart in enumerate(diet_charts) if chart.get("is_recommended")), 1
    )

    overview_data = [
        ["Plan Details", "Information"],
        ["Total Diet Charts", str(total_charts)],
        ["Recommended Chart", f"Chart #{recommended_chart}"],
        ["Average Daily Calories", f"{avg_calories:.0f} calories"],
        ["Calorie Range", f"{min_calories} - {max_calories} calories"],
        ["Plan Duration", "Daily meal plans"],
    ]

    overview_table = Table(overview_data, colWidths=[3.2 * inch, 2.8 * inch])
    overview_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), accent_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("BACKGROUND", (0, 1), (-1, -1), light_gray),
            ("GRID", (0, 0), (-1, -1), 1, border_color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])
    )

    story.append(KeepTogether([overview_table]))

    # ===== PAGE BREAK BEFORE DIET CHARTS =====
    story.append(PageBreak())

    # ===== INDIVIDUAL DIET CHARTS =====
    for i, chart in enumerate(diet_charts):
        # Start each chart on a new page
        if i > 0:
            story.append(PageBreak())

        # Chart header with better spacing
        chart_title = f"Diet Chart #{chart['chart_number']}"
        if chart.get("is_recommended"):
            chart_title += " - RECOMMENDED"

        story.append(Paragraph(chart_title, section_header_style))

        # Total calories info
        calories_info = ParagraphStyle(
            "CaloriesInfo",
            parent=normal_style,
            fontSize=13,
            fontName="Helvetica-Bold",
            textColor=accent_color,
            spaceAfter=15,
            spaceBefore=5,
        )
        story.append(
            Paragraph(
                f"Total Daily Calories: {chart['total_calories']} calories",
                calories_info,
            )
        )

        # Process each meal with better organization
        meal_sections = [
            ("breakfast", "Breakfast", warning_color),
            ("lunch", "Lunch", accent_color),
            ("snacks", "Snacks", success_color),
            ("dinner", "Dinner", primary_color),
        ]

        chart_elements = []  # Collect all elements for this chart
        
        for meal_key, meal_name, meal_color in meal_sections:
            meal_items = chart.get(meal_key, [])
            if not meal_items:
                continue

            # Calculate meal calories
            meal_calories = sum(item.get("calories", 0) for item in meal_items)

            # Meal header
            meal_title = f"{meal_name} ({meal_calories} calories)"
            meal_header_para = Paragraph(meal_title, meal_header_style)
            
            # Create meal items table with better sizing
            meal_table_data = [["Food Item", "Calories", "Notes"]]

            for item in meal_items:
                food_name = item.get("name", "Unknown")
                calories = item.get("calories", 0)
                notes = item.get("notes", item.get("description", ""))
                # Limit notes length to prevent overflow
                if len(notes) > 50:
                    notes = notes[:47] + "..."

                meal_table_data.append([food_name, f"{calories}", notes])

            meal_table = Table(
                meal_table_data, 
                colWidths=[3.0 * inch, 0.8 * inch, 2.2 * inch],
                repeatRows=1  # Repeat header if table splits
            )
            meal_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), meal_color),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("ALIGN", (0, 0), (0, -1), "LEFT"),
                    ("ALIGN", (1, 1), (1, -1), "CENTER"),
                    ("ALIGN", (2, 1), (2, -1), "LEFT"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, border_color),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    # Alternate row colors
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f8fafc")]),
                ])
            )

            # Keep meal header with its table
            chart_elements.append(KeepTogether([
                meal_header_para,
                Spacer(1, 5),
                meal_table,
                Spacer(1, 15)
            ]))

        # Add all chart elements
        story.extend(chart_elements)

    # ===== PAGE BREAK BEFORE RECOMMENDATIONS =====
    story.append(PageBreak())

    # ===== RECOMMENDATIONS SECTION =====
    story.append(Paragraph("Health & Dietary Recommendations", section_header_style))

    recommendations = get_recommendations_by_risk(outcome["risk_category"])

    recommendation_elements = []
    for i, recommendation in enumerate(recommendations, 1):
        rec_style = ParagraphStyle(
            "RecommendationStyle",
            parent=normal_style,
            leftIndent=15,
            bulletIndent=5,
            spaceAfter=10,
            spaceBefore=5,
        )
        recommendation_elements.append(
            Paragraph(f"{i}. {recommendation}", rec_style)
        )

    story.extend(recommendation_elements)
    story.append(Spacer(1, 0.3 * inch))

    # ===== IMPORTANT NOTES SECTION =====
    story.append(Paragraph("Important Guidelines", section_header_style))

    guidelines = [
        "Follow the recommended portion sizes for optimal results.",
        "Drink at least 8-10 glasses of water throughout the day.",
        "Maintain regular meal timings as suggested in the charts.",
        "Include physical activity as recommended by your healthcare provider.",
        "Monitor your body's response to the diet plan and adjust as needed.",
    ]

    guideline_elements = []
    for guideline in guidelines:
        guideline_style = ParagraphStyle(
            "GuidelineStyle",
            parent=normal_style,
            leftIndent=15,
            bulletIndent=5,
            spaceAfter=8,
            spaceBefore=4,
        )
        guideline_elements.append(
            Paragraph(f"• {guideline}", guideline_style)
        )

    story.extend(guideline_elements)
    story.append(Spacer(1, 0.3 * inch))

    # ===== DISCLAIMER SECTION =====
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=normal_style,
        fontSize=10,
        textColor=HexColor("#6b7280"),
        borderWidth=1,
        borderColor=border_color,
        borderPadding=15,
        backColor=HexColor("#fef2f2"),
        spaceAfter=20,
        spaceBefore=10,
        leading=14,
    )

    disclaimer_text = """
    <b>Medical Disclaimer:</b> This diet plan is generated based on risk assessment algorithms and is for 
    informational purposes only. It should not replace professional medical advice, diagnosis, or treatment. 
    Always consult with a qualified healthcare provider or registered dietitian before making significant 
    changes to your diet, especially if you have existing health conditions, allergies, or are taking medications.
    """

    story.append(Paragraph(disclaimer_text, disclaimer_style))

    # ===== FOOTER =====
    footer_style = ParagraphStyle(
        "Footer",
        parent=normal_style,
        fontSize=9,
        alignment=TA_CENTER,
        textColor=HexColor("#9ca3af"),
        spaceAfter=5,
        spaceBefore=10,
    )

    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "Generated by CarePulse - Your Digital Health Companion", footer_style
        )
    )
    story.append(
        Paragraph(f"Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}", footer_style)
    )

    # Build the PDF with better frame settings
    doc.build(story)
    return buffer

def get_recommendations_by_risk(risk_category):
    """Get detailed recommendations based on risk level"""
    if risk_category == "high_risk":
        return [
            "Follow a strict low-sodium diet with less than 2,300mg sodium per day",
            "Limit saturated fats to less than 10% of total daily calories",
            "Include omega-3 rich foods such as fatty fish, walnuts, and flaxseeds daily",
            "Consume at least 5 servings of fruits and vegetables per day",
            "Choose whole grains over refined grains in all meals",
            "Eliminate or severely limit processed foods, fast foods, and added sugars",
            "Practice strict portion control and consider using smaller plates",
            "Schedule regular follow-ups with your healthcare provider",
        ]
    elif risk_category == "moderate_risk":
        return [
            "Maintain a well-balanced diet with variety from all food groups",
            "Include heart-healthy fats like olive oil, avocados, and nuts daily",
            "Choose lean proteins such as poultry, fish, legumes, and tofu",
            "Consume high-fiber foods including vegetables, fruits, and whole grains",
            "Limit processed and fried foods to occasional treats",
            "Monitor portion sizes and practice mindful eating",
            "Stay adequately hydrated with water as your primary beverage",
            "Consider meal timing and avoid late-night eating",
        ]
    else:  # low_risk
        return [
            "Continue maintaining healthy eating patterns as prevention",
            "Include a colorful variety of fruits and vegetables in your daily meals",
            "Choose lean proteins and vary your protein sources throughout the week",
            "Stay physically active and maintain a healthy weight",
            "Practice good hydration habits with adequate water intake",
            "Use mindful eating practices and enjoy your meals",
            "Plan and prepare meals ahead of time when possible",
            "Schedule regular health check-ups to monitor your continued wellness",
        ]


# ===== HELPER FUNCTIONS FOR ENHANCED PDF WITH IMAGES =====


def create_compact_meal_table_style(meal_color, num_rows):
    """Create table style for compact meal tables"""
    style_list = [
        # Header styling
        ("BACKGROUND", (0, 0), (-1, 0), meal_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        # Data styling
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),  # Right align calories
        ("ALIGN", (0, 1), (0, -1), "CENTER"),  # Center align images
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # Tight padding for compact layout
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        # Row height constraints
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f8fafc")]),
    ]

    return TableStyle(style_list)


@app.route("/about")
def about():
    """About page"""
    return render_template("about.html")


@app.route("/contact")
def contact():
    """Contact page"""
    return render_template("contact.html")


@app.route("/health")
def health_check():
    """Health check endpoint"""
    global crawling_status
    status = {
        "status": "healthy",
        "model_loaded": bool(models and models.get("model")),
        "diet_plan_loaded": bool(diet_plan),
        "image_crawler": "Bing Image Crawler",
        "crawling_status": {
            "total_cached_images": len(
                [s for s in crawling_status.values() if s != "crawling"]
            ),
            "currently_crawling": len(
                [s for s in crawling_status.values() if s == "crawling"]
            ),
            "cache_size": len(crawling_status),
        },
        "timestamp": datetime.now().isoformat(),
    }
    return jsonify(status)


# Add favicon route to prevent 404 errors
@app.route("/favicon.ico")
def favicon():
    try:
        return send_file("static/favicon.ico", mimetype="image/vnd.microsoft.icon")
    except:
        return "", 204  # No content response if favicon not found


@app.errorhandler(404)
def not_found_error(error):
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Page Not Found - CarePulse</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                margin-top: 100px; 
                background-color: #f8f9fa;
            }
            .container {
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 { color: #dc3545; }
            .btn {
                display: inline-block;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>404 - Page Not Found</h1>
            <p>The page you are looking for doesn't exist.</p>
            <a href="/" class="btn">Return to Home</a>
            <a href="/predict" class="btn">Heart Disease Prediction</a>
        </div>
    </body>
    </html>
    """), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Server Error - CarePulse</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                margin-top: 100px; 
                background-color: #f8f9fa;
            }
            .container {
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 { color: #dc3545; }
            .btn {
                display: inline-block;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>500 - Internal Server Error</h1>
            <p>Something went wrong on our end. Please try again later.</p>
            <a href="/" class="btn">Return to Home</a>
        </div>
    </body>
    </html>
    """), 500


if __name__ == "__main__":
    print("=" * 60)
    print("🏥 Starting CarePulse Flask Application with Bing Image Crawler...")
    print("=" * 60)

    # Create placeholder image
    create_placeholder_image()

    # Check if required files exist
    required_files = [
        "models/heart_disease_model_20250906_075310.pkl",
        "data/Final_diet_plan.json",
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"✗ Missing: {file_path}")

    # Status summary
    print("\n📊 Application Status:")
    print(f"   Models loaded: {'✓ Yes' if models and models.get('model') else '✗ No'}")
    print(f"   Diet plan loaded: {'✓ Yes' if diet_plan else '✗ No'}")
    print(f"   Image crawler: ✓ Bing Image Crawler")

    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("   Please ensure all required files are in place.")
    else:
        print("\n✅ All required files found!")

    print(f"\n🚀 Server will start at: http://localhost:5000")
    print("   Available endpoints:")
    print("   - / (Home)")
    print("   - /predict (Heart Disease Prediction)")
    print("   - /diet-plan (Diet Plan Generation)")
    print("   - /health (Health Check)")
    print("   - /get-food-image/<food_name> (Food Image API)")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
