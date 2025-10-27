#!/usr/bin/env python3
"""
Cron job script to check data drift by calling the db_api_server API endpoint.
This script periodically calls /api/db_check_drift/ and logs the results.
"""

import requests
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path to import config
base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(base_abspath)

db_host = "localhost"
db_port = 8000

def load_config():
    """Load configuration from cron_config.yaml"""
    config_path = Path(__file__).parent / "cron_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_file: str, log_level: str = "INFO"):
    """Setup logging configuration"""
    log_path = Path(__file__).parent / log_file

    # Create logs directory if it doesn't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        filename=str(log_path),
        filemode='a',
        format='%(asctime)s [%(levelname)s]: %(message)s',
        level=level,
        force=True
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def call_drift_api(api_url: str, params: dict, timeout: int = 30):
    """
    Call the drift detection API endpoint.

    Args:
        api_url (str): Full URL to the API endpoint
        params (dict): Query parameters for the API call
        timeout (int): Request timeout in seconds

    Returns:
        dict: API response or error information
    """
    try:
        logging.info(f"Calling drift detection API: {api_url}")
        logging.info(f"Parameters: {json.dumps(params, indent=2)}")

        response = requests.get(api_url, params=params, timeout=timeout)
        response.raise_for_status()

        result = response.json()
        logging.info(f"API Response: {json.dumps(result, indent=2)}")

        return result

    except requests.exceptions.Timeout:
        error_msg = f"API request timed out after {timeout} seconds"
        logging.error(error_msg)
        return {"status": "error", "message": error_msg}

    except requests.exceptions.ConnectionError as e:
        error_msg = f"Failed to connect to API server: {str(e)}"
        logging.error(error_msg)
        return {"status": "error", "message": error_msg}

    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP error occurred: {str(e)}"
        logging.error(error_msg)
        return {"status": "error", "message": error_msg}

    except Exception as e:
        error_msg = f"Unexpected error calling API: {str(e)}"
        logging.error(error_msg)
        return {"status": "error", "message": error_msg}


# def process_drift_result(result: dict, alert_on_drift: bool = True):
#     """
#     Process the drift detection result and take appropriate actions.

#     Args:
#         result (dict): Result from the drift detection API
#         alert_on_drift (bool): Whether to trigger alerts on drift detection
#     """
#     if result.get("status") == "error":
#         logging.error(f"Drift check failed: {result.get('message')}")
#         return

#     if result.get("status") == "no_data":
#         logging.warning(f"No data available: {result.get('message')}")
#         return

#     if result.get("status") == "success":
#         drift_detected = result.get("drift_detected", False)
#         false_ratio = result.get("false_ratio", 0)
#         threshold = result.get("threshold", 0)
#         total_count = result.get("total_count", 0)
#         false_count = result.get("false_count", 0)

#         if drift_detected:
#             log_msg = (
#                 f"⚠️  DATA DRIFT DETECTED! "
#                 f"False ratio: {false_ratio:.1%} >= threshold: {threshold:.1%} "
#                 f"({false_count}/{total_count} records)"
#             )
#             logging.warning(log_msg)

#             if alert_on_drift:
#                 # Here you can add additional alert mechanisms:
#                 # - Send email
#                 # - Send Slack/Discord notification
#                 # - Trigger webhook
#                 # - Write to alert file
#                 alert_file = Path(__file__).parent / "drift_alerts.log"
#                 with alert_file.open("a", encoding="utf-8") as f:
#                     f.write(f"{datetime.now().isoformat()} - {log_msg}\n")
#                     f.write(f"  Details: {json.dumps(result, indent=2)}\n")

#         else:
#             log_msg = (
#                 f"✓ No drift detected. "
#                 f"False ratio: {false_ratio:.1%} < threshold: {threshold:.1%} "
#                 f"({false_count}/{total_count} records)"
#             )
#             logging.info(log_msg)


def main():
    """Main function for the cron job"""
    try:
        # Load configuration
        config = load_config()

        # Setup logging
        setup_logging(
            log_file=config.get("logging", {}).get("log_file", "drift_checker.log"),
            log_level=config.get("logging", {}).get("log_level", "INFO")
        )

        logging.info("=" * 60)
        logging.info("Starting drift detection check")
        logging.info("=" * 60)

        # # Get API configuration
        # api_config = config.get("api", {})
        # db_host = api_config.get("host", "localhost")
        # db_port = api_config.get("port", 8000)
        # endpoint = api_config.get("endpoint", "/api/db_check_drift/")

        drift_api_url = f"http://{db_host}:{db_port}/api/db_check_drift/"
        # Get drift check parameters
        # params = config.get("drift_params", {})
        # timeout = api_config.get("timeout", 30)
        # Get drift check parameters from config or use defaults
        params = {
            "period": "1 day",           # 또는 config에서 가져오기
            "class_name": "person",
            "threshold": 0.3              # 또는 config에서 가져오기
        }
        timeout = 30
        # Call the API
        result = call_drift_api(drift_api_url, params, timeout)
        print(result)
        drift_detected = None
        
        retrain_api_url = f"http://{db_host}:{db_port}/api/db_retrain/"

        if result.get("status") == "success":
            drift_detected = result.get("drift_detected", False)

            if True==drift_detected:
                # request.get(retrain_api_url,...) # trigger retrain host = "localhost" port = 8000 endpoint = "/api/db_retrain/"
                pass

        # # Process the result
        # alert_on_drift = config.get("alert", {}).get("enabled", True)
        # process_drift_result(result, alert_on_drift)

        logging.info("Drift detection check completed")
        logging.info("=" * 60)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
