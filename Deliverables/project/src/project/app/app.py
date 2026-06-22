"""
This script launches a Flask web application to serve a real-time anomaly
detection dashboard.

The application simulates a stream of production data, processes it with a
pre-trained machine learning model, and visualizes the data and anomaly
predictions on a web-based dashboard. It is designed to demonstrate the
functionality of the `ProductionAnomalyPredictor` in a practical, interactive
setting.

Key functionalities include:

- Loading historical and test data from CSV files.
- Serving a main dashboard page (`dashboard.html`).
- Providing API endpoints for the frontend to fetch initial data, request
  subsequent data points with AI predictions, and control the simulation
  (play, pause, reset).
- Lazy-loading the machine learning model to reduce initial startup time.
- A custom JSON encoder to handle NumPy data types.
"""
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask.json.provider import DefaultJSONProvider
from datetime import datetime
import json
import warnings
import os
import sys  # Import sys to check for Sphinx

# Suppress potential scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    # Corrected import for src-layout projects
    from src.project.models.predict import ProductionAnomalyPredictor
except ImportError as e:
    # If the import fails, print a fatal error and exit, as the app cannot run.
    print(f"FATAL ERROR: Could not import ProductionAnomalyPredictor. Details: {e}")
    exit()


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle serialization of NumPy data types."""

    def default(self, obj):
        """
        Override the default method to handle specific data types.

        Args:
            obj: The object to encode.

        Returns:
            A serializable representation of the object.
        """
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyJSONEncoder, self).default(obj)


def _numpy_default(obj):
    """Convert NumPy and datetime values into JSON-serializable types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return DefaultJSONProvider.default(obj)


class NumpyJSONProvider(DefaultJSONProvider):
    """Flask 3 JSON provider that serializes NumPy and datetime values.

    Flask 3 removed app.json_encoder, so a JSON provider is used to make the
    NumPy handling actually take effect on jsonify responses.
    """

    default = staticmethod(_numpy_default)


# Initialize the Flask application
app = Flask(__name__)
# Register the custom JSON provider so jsonify can serialize NumPy types.
app.json = NumpyJSONProvider(app)
# Kept for backwards compatibility with code/tests that reference the encoder.
app.json_encoder = NumpyJSONEncoder

# --- Use repo-relative paths so the app does not depend on the working directory ---
basedir = os.path.abspath(os.path.dirname(__file__))

# --- Configuration ---
TRAIN_DATA_PATH = os.path.join(basedir, "BKNG_engineering.csv")
TEST_DATA_PATH = os.path.join(basedir, "BKNG_engineering_test.csv")
# Model artifacts live in ../models/saved_models relative to this file.
MODELS_PATH = os.path.join(basedir, "../models/saved_models/")
HISTORICAL_POINTS_TO_LOAD = 50000


# --- Data Loading Section ---
# Define default empty values for the dataframes.
train_data = pd.DataFrame()
test_data = pd.DataFrame()
test_data_count = 0

# A more robust check to see if Sphinx is running
is_sphinx_build = 'sphinx' in sys.modules

if not is_sphinx_build:
    try:
        print("--- Reading and Preparing Data Files ---")
        full_train_df = pd.read_csv(
            TRAIN_DATA_PATH, usecols=["Date", "Close", "event"], parse_dates=["Date"]
        )
        test_data = pd.read_csv(TEST_DATA_PATH, parse_dates=["Date"])
        test_data_count = len(test_data)
        print(f"Successfully loaded {test_data_count} points from the test data file.")

        if not test_data.empty:
            test_start_date = test_data["Date"].iloc[0]
            historical_data_for_chart = full_train_df[
                full_train_df["Date"] < test_start_date
            ].copy()
        else:
            historical_data_for_chart = full_train_df

        train_data = historical_data_for_chart.tail(HISTORICAL_POINTS_TO_LOAD).copy()
        train_data["Date"] = train_data["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"Prepared {len(train_data)} points of historical data for the chart.")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: Data file not found. Ensure '{e.filename}' exists.")
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")


# --- Predictor and Simulation State ---
predictor = None
current_index = 0
is_paused = True


# --- Flask Routes ---
@app.route("/")
def dashboard():
    """
    Serves the main dashboard.html page.

    This is the entry point for the user's web browser.

    Returns:
        The rendered dashboard.html template.
    """
    return render_template("dashboard.html")


@app.route("/api/init_data")
def get_init_data():
    """
    Provides the initial data needed by the frontend to set up the dashboard.

    This endpoint is called once when the webpage loads. It resets the simulation
    state and provides the historical data for the chart background and the very
    first point of the test data.

    Returns:
        A JSON object containing:
        - "train_data": A list of historical data points.
        - "initial_test_point": The first data point from the test set.
        - "test_data_count": The total number of points in the test set.
    """
    global current_index
    current_index = 0
    if predictor:
        predictor.reset_buffer()

    if not test_data.empty:
        first_test_point = test_data.iloc[0].to_dict()
    else:
        first_test_point = None

    return jsonify(
        {
            "train_data": train_data.to_dict("records"),
            "initial_test_point": first_test_point,
            "test_data_count": test_data_count,
        }
    )


@app.route("/api/next_point")
def get_next_point():
    """
    Provides the next data point from the test set and its AI-driven prediction.

    This endpoint is called repeatedly by the frontend to drive the simulation.
    It handles lazy-loading of the ML model on the first call. It checks the
    simulation state (paused, running, complete) and returns the appropriate
    response.

    Returns:
        A JSON object with a "status" key.

        - If "status" is "paused", no data is returned.
        - If "status" is "complete", it signals the end of the simulation.
        - If "status" is "running", it includes the current data point, the
          model's prediction, and the current index.
        - If "status" is "error", it includes an error message.
    """
    global current_index, predictor

    if is_paused:
        return jsonify({"status": "paused"})

    if current_index >= test_data_count:
        return jsonify({"status": "complete"})

    if predictor is None:
        print("\n--- LAZY LOADING PREDICTOR MODEL ---")
        try:
            predictor = ProductionAnomalyPredictor(
                models_path=MODELS_PATH
            )
            print("--- PREDICTOR LOADED SUCCESSFULLY ---\n")
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to load model: {e}"})

    current_point_series = test_data.iloc[current_index]
    current_point = current_point_series.to_dict()
    prediction = predictor.predict_single_point(current_point)
    prediction["explicit_label"] = "AI-Generated Prediction"

    response = {
        "status": "running",
        "data": current_point,
        "prediction": prediction,
        "index": current_index,
    }

    current_index += 1
    return jsonify(response)


@app.route("/api/control", methods=["POST"])
def control_simulation():
    """
    Handles simulation control commands (play, pause, reset) from the frontend.

    This endpoint receives a POST request with a JSON payload specifying the
    action to perform.

    Request JSON payload:
        {"action": "play" | "pause" | "reset"}

    Returns:
        A JSON object confirming the action and returning the current simulation state.
    """
    global current_index, is_paused

    data = request.get_json()
    action = data.get("action")

    if action == "play":
        is_paused = False
    elif action == "pause":
        is_paused = True
    elif action == "reset":
        current_index = 0
        is_paused = True
        if predictor:
            predictor.reset_buffer()

    return jsonify({"success": True, "state": "paused" if is_paused else "running"})


if __name__ == "__main__":
    """
    Main entry point for the script.
    
    Runs the Flask application in debug mode, making it accessible on the
    local network. This block is ignored when the file is imported by Sphinx.
    """
    app.run(host="0.0.0.0", port=5000, debug=True)