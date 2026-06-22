import sys
import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
from unittest.mock import Mock, patch, mock_open, MagicMock
from datetime import datetime
from flask import Flask

# Import the application components
from src.project.app.app import (
    app,
    NumpyJSONEncoder,
)


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture
def sample_train_data():
    """Sample training data for testing."""
    dates = pd.date_range("2024-01-01 09:30:00", periods=100, freq="1min")
    return pd.DataFrame(
        {
            "Date": dates,
            "Close": np.random.uniform(100, 200, 100),
            "event": ["normal"] * 80 + ["dip"] * 10 + ["crash"] * 10,
        }
    )


@pytest.fixture
def sample_test_data():
    """Sample test data for testing."""
    dates = pd.date_range("2024-01-02 09:30:00", periods=50, freq="1min")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.random.uniform(100, 200, 50),
            "High": np.random.uniform(150, 250, 50),
            "Low": np.random.uniform(50, 150, 50),
            "Close": np.random.uniform(100, 200, 50),
            "Volume": np.random.randint(1000, 100000, 50),
            "RSI_14": np.random.uniform(20, 80, 50),
            "MACD": np.random.uniform(-2, 2, 50),
            "event": ["normal"] * 40 + ["dip"] * 10,
        }
    )


@pytest.fixture
def mock_predictor():
    """Mock predictor for testing."""
    predictor = Mock()
    predictor.predict_single_point.return_value = {
        "status": "success",
        "is_anomaly": False,
        "anomaly_probability": 0.3,
        "predicted_anomaly_type": "normal",
        "confidence": 0.7,
        "type_probabilities": None,
        "timestamp": "2024-01-02T09:30:00",
    }
    predictor.reset_buffer.return_value = None
    return predictor


class TestNumpyJSONEncoder:
    """Test suite for NumpyJSONEncoder class."""

    def test_numpy_integer_encoding(self):
        """Test encoding of NumPy integers."""
        encoder = NumpyJSONEncoder()

        # Test specific NumPy integer types (not abstract base class)
        assert encoder.default(np.int64(42)) == 42
        assert encoder.default(np.int32(100)) == 100
        # FIXED: Test with concrete numpy integer types, not abstract np.integer
        assert encoder.default(np.int16(50)) == 50

    def test_numpy_float_encoding(self):
        """Test encoding of NumPy floats."""
        encoder = NumpyJSONEncoder()

        # Test various NumPy float types
        assert encoder.default(np.float64(3.14)) == 3.14
        # FIXED: Use pytest.approx for float32 precision issues
        assert encoder.default(np.float32(2.71)) == pytest.approx(2.71, rel=1e-5)
        # FIXED: Test with concrete numpy float types, not abstract np.floating
        assert encoder.default(np.float16(1.23)) == pytest.approx(1.23, rel=1e-2)

    def test_numpy_array_encoding(self):
        """Test encoding of NumPy arrays."""
        encoder = NumpyJSONEncoder()

        arr = np.array([1, 2, 3, 4, 5])
        assert encoder.default(arr) == [1, 2, 3, 4, 5]

        # Test 2D array
        arr_2d = np.array([[1, 2], [3, 4]])
        assert encoder.default(arr_2d) == [[1, 2], [3, 4]]

    def test_numpy_bool_encoding(self):
        """Test encoding of NumPy booleans."""
        encoder = NumpyJSONEncoder()

        assert encoder.default(np.bool_(True)) is True
        assert encoder.default(np.bool_(False)) is False

    def test_datetime_encoding(self):
        """Test encoding of datetime objects."""
        encoder = NumpyJSONEncoder()

        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = encoder.default(dt)
        assert result == "2024-01-15T10:30:00"

    def test_unsupported_type_encoding(self):
        """Test that unsupported types raise TypeError."""
        encoder = NumpyJSONEncoder()

        class CustomObject:
            pass

        with pytest.raises(TypeError):
            encoder.default(CustomObject())

    def test_json_serialization_integration(self):
        """Test integration with json.dumps()."""
        data = {
            "integer": np.int64(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
            "bool": np.bool_(True),
            "datetime": datetime(2024, 1, 15, 10, 30, 0),
        }

        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert parsed["integer"] == 42
        assert parsed["float"] == 3.14
        assert parsed["array"] == [1, 2, 3]
        assert parsed["bool"] is True
        assert parsed["datetime"] == "2024-01-15T10:30:00"


class TestFlaskRoutes:
    """Test suite for Flask application routes."""

    def test_dashboard_route(self, client):
        """Test the main dashboard route."""
        with patch("src.project.app.app.render_template") as mock_render:
            mock_render.return_value = "<html>Dashboard</html>"

            response = client.get("/")

            assert response.status_code == 200
            mock_render.assert_called_once_with("dashboard.html")


class TestNextPointRoute:
    """Test suite for /api/next_point route."""

    def test_next_point_when_paused(self, client):
        """Test next point route when simulation is paused."""
        import src.project.app.app as app_module

        original_is_paused = app_module.is_paused

        try:
            app_module.is_paused = True
            response = client.get("/api/next_point")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "paused"
        finally:
            app_module.is_paused = original_is_paused

    def test_next_point_when_complete(self, client):
        """Test next point route when simulation is complete."""
        import src.project.app.app as app_module

        original_current_index = app_module.current_index
        original_test_data_count = app_module.test_data_count
        original_is_paused = app_module.is_paused

        try:
            app_module.current_index = 100
            app_module.test_data_count = 50
            app_module.is_paused = False

            response = client.get("/api/next_point")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "complete"
        finally:
            app_module.current_index = original_current_index
            app_module.test_data_count = original_test_data_count
            app_module.is_paused = original_is_paused

    @patch("src.project.models.predict.ProductionAnomalyPredictor")
    def test_next_point_lazy_load_predictor_success(
        self, mock_predictor_class, client, mock_predictor
    ):
        """Test successful lazy loading of predictor."""
        import src.project.app.app as app_module

        mock_predictor_class.return_value = mock_predictor

        # Store original values
        original_predictor = app_module.predictor
        original_is_paused = app_module.is_paused
        original_current_index = app_module.current_index
        original_test_data_count = app_module.test_data_count
        original_test_data = app_module.test_data

        try:
            # Set up test scenario
            app_module.predictor = None
            app_module.is_paused = False
            app_module.current_index = 0
            app_module.test_data_count = 50

            # Mock test data
            mock_test_data = Mock()
            mock_test_data.iloc = [Mock()]
            mock_test_data.iloc[0].to_dict.return_value = {
                "Date": "2024-01-02T09:30:00",
                "Close": 150.0,
            }
            app_module.test_data = mock_test_data

            response = client.get("/api/next_point")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "running"
            assert "data" in data
            assert "prediction" in data
            assert "index" in data
        finally:
            # Restore original values
            app_module.predictor = original_predictor
            app_module.is_paused = original_is_paused
            app_module.current_index = original_current_index
            app_module.test_data_count = original_test_data_count
            app_module.test_data = original_test_data

    def test_next_point_with_predictor_loaded(self, client, mock_predictor):
        """Test next point route with predictor already loaded."""
        import src.project.app.app as app_module

        # Store original values
        original_predictor = app_module.predictor
        original_is_paused = app_module.is_paused
        original_current_index = app_module.current_index
        original_test_data_count = app_module.test_data_count
        original_test_data = app_module.test_data

        try:
            # Set up test scenario
            app_module.predictor = mock_predictor
            app_module.is_paused = False
            app_module.current_index = 0
            app_module.test_data_count = 50

            # Setup test data mock
            mock_point = {
                "Date": "2024-01-02T09:30:00",
                "Close": 150.0,
                "Volume": 10000,
                "RSI_14": 55.0,
            }

            mock_test_data = Mock()
            mock_test_data.iloc = [Mock()]
            mock_test_data.iloc[0].to_dict.return_value = mock_point
            app_module.test_data = mock_test_data

            response = client.get("/api/next_point")

            assert response.status_code == 200
            data = response.get_json()

            assert data["status"] == "running"
            assert data["data"] == mock_point
            assert "prediction" in data
            assert data["prediction"]["explicit_label"] == "AI-Generated Prediction"
            assert data["index"] == 0

            # Verify predictor was called
            mock_predictor.predict_single_point.assert_called_once_with(mock_point)
        finally:
            # Restore original values
            app_module.predictor = original_predictor
            app_module.is_paused = original_is_paused
            app_module.current_index = original_current_index
            app_module.test_data_count = original_test_data_count
            app_module.test_data = original_test_data


class TestControlRoute:
    """Test suite for /api/control route."""

    def test_control_play_action(self, client):
        """Test play action in control route."""
        import src.project.app.app as app_module

        original_is_paused = app_module.is_paused

        try:
            app_module.is_paused = True
            response = client.post("/api/control", json={"action": "play"})

            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True
            assert data["state"] == "running"
        finally:
            app_module.is_paused = original_is_paused

    def test_control_pause_action(self, client):
        """Test pause action in control route."""
        import src.project.app.app as app_module

        original_is_paused = app_module.is_paused

        try:
            app_module.is_paused = False
            response = client.post("/api/control", json={"action": "pause"})

            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True
            assert data["state"] == "paused"
        finally:
            app_module.is_paused = original_is_paused

    def test_control_reset_action(self, client, mock_predictor):
        """Test reset action in control route."""
        import src.project.app.app as app_module

        original_current_index = app_module.current_index
        original_is_paused = app_module.is_paused
        original_predictor = app_module.predictor

        try:
            app_module.current_index = 25
            app_module.is_paused = False
            app_module.predictor = mock_predictor

            response = client.post("/api/control", json={"action": "reset"})

            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True
            assert data["state"] == "paused"

            # Verify predictor buffer was reset
            mock_predictor.reset_buffer.assert_called_once()
        finally:
            app_module.current_index = original_current_index
            app_module.is_paused = original_is_paused
            app_module.predictor = original_predictor

    def test_control_reset_action_no_predictor(self, client):
        """Test reset action when predictor is None."""
        import src.project.app.app as app_module

        original_current_index = app_module.current_index
        original_is_paused = app_module.is_paused
        original_predictor = app_module.predictor

        try:
            app_module.current_index = 25
            app_module.is_paused = False
            app_module.predictor = None

            response = client.post("/api/control", json={"action": "reset"})

            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True
            assert data["state"] == "paused"
        finally:
            app_module.current_index = original_current_index
            app_module.is_paused = original_is_paused
            app_module.predictor = original_predictor

    def test_control_invalid_action(self, client):
        """Test control route with invalid action."""
        response = client.post("/api/control", json={"action": "invalid_action"})

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

    def test_control_missing_action(self, client):
        """Test control route with missing action."""
        response = client.post("/api/control", json={})

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True


class TestDataLoading:
    """Test suite for data loading functionality."""

    @patch("pandas.read_csv")
    @patch("builtins.print")
    def test_successful_data_loading(
        self, mock_print, mock_read_csv, sample_train_data, sample_test_data
    ):
        """Test successful loading of data files."""
        mock_read_csv.side_effect = [sample_train_data, sample_test_data]

        # Re-import to trigger data loading
        with patch.dict("sys.modules"):
            if "src.project.app.app" in sys.modules:
                del sys.modules["src.project.app.app"]

            import src.project.app.app as test_app

            # Verify CSV reading was called
            assert mock_read_csv.call_count == 2

            # Verify success messages were printed
            mock_print.assert_any_call("--- Reading and Preparing Data Files ---")

    @patch("pandas.read_csv")
    @patch("builtins.print")
    def test_file_not_found_error(self, mock_print, mock_read_csv):
        """Test handling of FileNotFoundError during data loading."""
        # FIXED: Create FileNotFoundError with filename attribute
        error = FileNotFoundError("test_file.csv")
        error.filename = "test_file.csv"
        mock_read_csv.side_effect = error

        # Re-import to trigger data loading
        with patch.dict("sys.modules"):
            if "src.project.app.app" in sys.modules:
                del sys.modules["src.project.app.app"]

            import src.project.app.app as test_app

            # FIXED: Check for the actual error message format
            mock_print.assert_any_call(
                "FATAL ERROR: Data file not found. Ensure 'test_file.csv' exists."
            )

    @patch("pandas.read_csv")
    @patch("builtins.print")
    def test_general_data_loading_error(self, mock_print, mock_read_csv):
        """Test handling of general errors during data loading."""
        mock_read_csv.side_effect = Exception("General error")

        # Re-import to trigger data loading
        with patch.dict("sys.modules"):
            if "src.project.app.app" in sys.modules:
                del sys.modules["src.project.app.app"]

            import src.project.app.app as test_app

            # Verify error message was printed
            mock_print.assert_any_call(
                "An unexpected error occurred during data loading: General error"
            )


class TestIntegration:
    """Integration tests for the complete Flask application."""

    def test_json_encoder_integration_with_flask(self, client):
        """Test that the custom JSON encoder works with Flask responses."""
        test_data = {
            "numpy_int": np.int64(42),
            "numpy_float": np.float64(3.14),
            "numpy_array": np.array([1, 2, 3]),
            "datetime": datetime(2024, 1, 15, 10, 30, 0),
        }

        with app.app_context():
            response = app.response_class(
                response=json.dumps(test_data, cls=NumpyJSONEncoder),
                status=200,
                mimetype="application/json",
            )

            data = json.loads(response.get_data(as_text=True))
            assert data["numpy_int"] == 42
            assert data["numpy_float"] == 3.14
            assert data["numpy_array"] == [1, 2, 3]
            assert data["datetime"] == "2024-01-15T10:30:00"


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_malformed_control_request(self, client):
        """Test control route with malformed request data."""
        # Test with string instead of JSON
        response = client.post("/api/control", data="not json")

        # FIXED: Flask returns 415 UNSUPPORTED MEDIA TYPE for malformed JSON
        assert response.status_code == 415

    def test_concurrent_requests_state_management(self, client, mock_predictor):
        """Test state management with concurrent-like requests."""
        import src.project.app.app as app_module

        # Store original values
        original_predictor = app_module.predictor
        original_test_data = app_module.test_data
        original_current_index = app_module.current_index
        original_is_paused = app_module.is_paused
        original_test_data_count = app_module.test_data_count

        try:
            app_module.predictor = mock_predictor

            mock_test_data = Mock()
            mock_test_data.iloc = [Mock()]
            mock_test_data.iloc[0].to_dict.return_value = {
                "Date": "2024-01-02T09:30:00",
                "Close": 150.0,
            }
            app_module.test_data = mock_test_data

            # Simulate state changes between requests
            app_module.current_index = 0
            app_module.is_paused = False
            app_module.test_data_count = 1

            # First request
            response1 = client.get("/api/next_point")
            assert response1.status_code == 200

            # Reset and try again
            client.post("/api/control", json={"action": "reset"})

            # Second request after reset
            response2 = client.get("/api/next_point")
            assert response2.status_code == 200
        finally:
            # Restore original values
            app_module.predictor = original_predictor
            app_module.test_data = original_test_data
            app_module.current_index = original_current_index
            app_module.is_paused = original_is_paused
            app_module.test_data_count = original_test_data_count


class TestConfiguration:
    """Test configuration and setup."""

    def test_app_configuration(self):
        """Test Flask app configuration."""
        assert app.json_encoder == NumpyJSONEncoder
        assert hasattr(app, "config")

    @patch("os.path.abspath")
    @patch("os.path.dirname")
    @patch("os.path.join")
    def test_path_configuration(self, mock_join, mock_dirname, mock_abspath):
        """Test that paths are configured correctly."""
        mock_abspath.return_value = "/app/path"
        mock_dirname.return_value = "/app"
        mock_join.side_effect = lambda base, file: f"{base}/{file}"

        # Re-import to test path configuration
        with patch.dict("sys.modules"):
            if "src.project.app.app" in sys.modules:
                del sys.modules["src.project.app.app"]

            import src.project.app.app as test_app

            # Verify path functions were called
            mock_abspath.assert_called()
            mock_dirname.assert_called()
            assert mock_join.call_count >= 2  # For TRAIN_DATA_PATH and TEST_DATA_PATH


class TestConstants:
    """Test application constants."""

    def test_constants_exist(self):
        """Test that required constants are defined."""
        from src.project.app.app import HISTORICAL_POINTS_TO_LOAD

        assert HISTORICAL_POINTS_TO_LOAD == 50000
        assert isinstance(HISTORICAL_POINTS_TO_LOAD, int)
        assert HISTORICAL_POINTS_TO_LOAD > 0
