#!/usr/bin/env python3
"""
Check data drift by calling the db_api_server API endpoint.
This script performs a single drift detection run.
"""

import requests
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path
import sys
import os
from email_alert import send_drift_alert_email

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
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        filename=str(log_path),
        filemode='a',
        format='%(asctime)s [%(levelname)s]: %(message)s',
        level=level,
        force=True
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def call_drift_api(api_url: str, params: dict, timeout: int = 30):
    """Call the drift detection API endpoint."""
    try:
        logging.info(f"Calling drift detection API: {api_url}")
        logging.info(f"Parameters: {json.dumps(params, indent=2)}")

        response = requests.get(api_url, params=params, timeout=timeout)
        response.raise_for_status()

        result = response.json()
        logging.info(f"API Response: {json.dumps(result, indent=2)}")
        return result

    except requests.exceptions.Timeout:
        msg = f"API request timed out after {timeout} seconds"
        logging.error(msg)
        return {"status": "error", "message": msg}

    # except requests.exceptions.ConnectionError as e:
    #     msg = f"Failed to connect to API server: {e}"
    #     logging.error(msg)
    #     return {"status": "error", "message": msg}

    # except requests.exceptions.HTTPError as e:
    #     msg = f"HTTP error occurred: {e}"
    #     logging.error(msg)
    #     return {"status": "error", "message": msg}

    # except Exception as e:
    #     msg = f"Unexpected error calling API: {e}"
    #     logging.error(msg)
    #     return {"status": "error", "message": msg}


def main():
    """Main drift detection process"""
    try:
        config = load_config()

        setup_logging(
            log_file=config.get("logging", {}).get("log_file", "logs/drift_checker.log"),
            log_level=config.get("logging", {}).get("log_level", "INFO")
        )

        logging.info("=" * 60)
        logging.info("Starting drift detection check")
        logging.info("=" * 60)

        drift_api_url = f"http://{db_host}:{db_port}/api/db_check_drift/"
        retrain_api_url = f"http://{db_host}:{db_port}/api/db_retrain/"

        params = {
            "period": "1 day",
            "class_name": "person",
            "threshold": 0.03
        }

        result = call_drift_api(drift_api_url, params)
        midlog_path = Path(__file__).parent / "logs/drift_midlog.jsonl"
        midlog_path.parent.mkdir(parents=True, exist_ok=True)

        # Save intermediate log
        with midlog_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "result": result
            }, ensure_ascii=False) + "\n")

        # If drift detected → trigger retraining
        if result.get("status") == "success" and result.get("drift_detected", False):
            logging.warning("⚠️ Drift detected! Triggering retrain process...")
            call_drift_api(retrain_api_url, {})
            # Send email alert if enabled
            if config.get('alert', {}).get('email', {}).get('enabled', False):
                try:
                    # Prepare drift information for email
                    drift_info = {
                        'drift_score': result.get('drift_score'),
                        'threshold': result.get('threshold'),
                        'features_affected': result.get('features_affected', []),
                        'timestamp': result.get('timestamp'),
                    }
                    
                    # Send email alert
                    email_config = config['alert']['email']
                    email_sent = send_drift_alert_email(email_config, drift_info)
                    
                    if email_sent:
                        logging.info("✅ Email alert sent successfully")
                    else:
                        logging.warning("⚠️ Failed to send email alert")
                        
                except Exception as e:
                    logging.error(f"Error sending email alert: {str(e)}")
            else:
                logging.info("Email alerts are disabled in configuration")

        logging.info("Drift detection check completed")
        logging.info("=" * 60)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    except Exception as e:
        logging.error(f"Unexpected error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
