"""
Email Alert Module for Drift Detection
Sends email notifications when model drift is detected
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional


class EmailAlert:
    """Handles email notifications for drift detection"""
    
    def __init__(self, config: Dict):
        """
        Initialize email alert system
        
        Args:
            config: Dictionary containing email configuration
        """
        self.enabled = config.get('enabled', False)
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.from_email = config.get('from_email')
        self.to_emails = config.get('to_emails', [])
        self.username = config.get('username', self.from_email)
        self.password = config.get('password')
        
    def send_drift_alert(self, drift_info: Dict) -> bool:
        """
        Send email alert when drift is detected
        
        Args:
            drift_info: Dictionary containing drift detection results
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not self.enabled:
            logging.info("Email alerts disabled - skipping notification")
            return False
            
        if not self._validate_config():
            logging.error("Invalid email configuration - cannot send alert")
            return False
        
        try:
            # Create email message
            msg = self._create_drift_message(drift_info)
            
            # Send email
            self._send_email(msg)
            
            logging.info(f"✅ Drift alert email sent to {', '.join(self.to_emails)}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send drift alert email: {str(e)}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate that all required email configuration is present"""
        required_fields = [
            self.smtp_server,
            self.smtp_port,
            self.from_email,
            self.to_emails,
            self.password
        ]
        
        if not all(required_fields):
            logging.error("Missing required email configuration fields")
            return False
            
        if not self.to_emails:
            logging.error("No recipient emails specified")
            return False
            
        return True
    
    def _create_drift_message(self, drift_info: Dict) -> MIMEMultipart:
        """
        Create formatted email message for drift alert
        
        Args:
            drift_info: Dictionary containing drift detection results
            
        Returns:
            MIMEMultipart: Formatted email message
        """
        msg = MIMEMultipart('alternative')
        msg['Subject'] = '⚠️ Model Drift Detected - Action Required'
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        
        # Extract drift information
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        drift_score = drift_info.get('drift_score', 'N/A')
        threshold = drift_info.get('threshold', 'N/A')
        features_affected = drift_info.get('features_affected', [])
        
        # Create plain text version
        text_body = f"""
DRIFT ALERT
===========

Model drift has been detected and requires attention.

Detection Time: {timestamp}
Drift Score: {drift_score}
Threshold: {threshold}
Status: DRIFT DETECTED

Features Affected:
{self._format_features_list(features_affected, plain_text=True)}

Action Taken:
- Automatic retraining has been triggered
- Model will be updated once retraining completes

Please review the drift report and monitor the retraining process.

---
This is an automated alert from the Model Drift Detection System
"""
        
        # Create HTML version
        html_body = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #ff6b6b; color: white; padding: 20px; border-radius: 5px; }}
        .content {{ background-color: #f9f9f9; padding: 20px; margin-top: 20px; border-radius: 5px; }}
        .info-row {{ margin: 10px 0; }}
        .label {{ font-weight: bold; color: #555; }}
        .value {{ color: #333; }}
        .features {{ background-color: white; padding: 15px; margin: 15px 0; border-left: 4px solid #ff6b6b; }}
        .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #777; }}
        ul {{ margin: 10px 0; padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>⚠️ Model Drift Alert</h2>
            <p>Drift has been detected and requires attention</p>
        </div>
        
        <div class="content">
            <div class="info-row">
                <span class="label">Detection Time:</span>
                <span class="value">{timestamp}</span>
            </div>
            <div class="info-row">
                <span class="label">Drift Score:</span>
                <span class="value">{drift_score}</span>
            </div>
            <div class="info-row">
                <span class="label">Threshold:</span>
                <span class="value">{threshold}</span>
            </div>
            <div class="info-row">
                <span class="label">Status:</span>
                <span class="value" style="color: #ff6b6b; font-weight: bold;">DRIFT DETECTED</span>
            </div>
            
            <div class="features">
                <h3>Features Affected:</h3>
                {self._format_features_list(features_affected, plain_text=False)}
            </div>
            
            <div style="background-color: #e8f5e9; padding: 15px; border-left: 4px solid #4caf50; margin-top: 15px;">
                <h3 style="margin-top: 0;">Action Taken:</h3>
                <ul>
                    <li>Automatic retraining has been triggered</li>
                    <li>Model will be updated once retraining completes</li>
                </ul>
            </div>
            
            <p style="margin-top: 20px;">
                Please review the drift report and monitor the retraining process.
            </p>
        </div>
        
        <div class="footer">
            This is an automated alert from the Model Drift Detection System
        </div>
    </div>
</body>
</html>
"""
        
        # Attach both versions
        part1 = MIMEText(text_body, 'plain')
        part2 = MIMEText(html_body, 'html')
        
        msg.attach(part1)
        msg.attach(part2)
        
        return msg
    
    def _format_features_list(self, features: List[str], plain_text: bool = True) -> str:
        """Format the list of affected features"""
        if not features:
            return "No specific features identified" if plain_text else "<p>No specific features identified</p>"
        
        if plain_text:
            return '\n'.join([f"  - {feature}" for feature in features])
        else:
            items = ''.join([f"<li>{feature}</li>" for feature in features])
            return f"<ul>{items}</ul>"
    
    def _send_email(self, msg: MIMEMultipart) -> None:
        """
        Send the email message via SMTP
        
        Args:
            msg: The email message to send
        """
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(self.username, self.password)
            server.send_message(msg)


def send_drift_alert_email(config: Dict, drift_info: Dict) -> bool:
    """
    Convenience function to send drift alert email
    
    Args:
        config: Email configuration dictionary
        drift_info: Drift detection results
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    alert = EmailAlert(config)
    return alert.send_drift_alert(drift_info)
