"""
Test script for email alert functionality
Run this to verify your email configuration works correctly
"""

import yaml
import logging
from email_alert import send_drift_alert_email

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_email_alert():
    """Test the email alert system"""
    
    print("\n" + "="*60)
    print("Testing Email Alert System")
    print("="*60 + "\n")
    
    # Load configuration
    try:
        with open('cron_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully")
    except FileNotFoundError:
        print("❌ Error: cron_config.yaml not found")
        return
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML: {e}")
        return
    
    # Check if email is enabled
    email_config = config.get('alert', {}).get('email', {})
    if not email_config.get('enabled', False):
        print("⚠️ Email alerts are DISABLED in configuration")
        print("   Set 'alert.email.enabled: true' to enable")
        return
    
    print(f"✅ Email alerts enabled")
    print(f"   SMTP Server: {email_config.get('smtp_server')}")
    print(f"   From: {email_config.get('from_email')}")
    print(f"   To: {', '.join(email_config.get('to_emails', []))}")
    print()
    
    # Create test drift information
    test_drift_info = {
        'drift_score': 0.85,
        'threshold': 0.70,
        'features_affected': [
            'feature_age',
            'feature_income',
            'feature_credit_score'
        ],
        'timestamp': '2025-10-29 10:30:00'
    }
    
    print("📧 Sending test email...")
    print(f"   Drift Score: {test_drift_info['drift_score']}")
    print(f"   Threshold: {test_drift_info['threshold']}")
    print(f"   Features Affected: {len(test_drift_info['features_affected'])}")
    print()
    
    # Send test email
    try:
        success = send_drift_alert_email(email_config, test_drift_info)
        
        if success:
            print("="*60)
            print("✅ SUCCESS! Test email sent successfully")
            print("="*60)
            print("\nPlease check your inbox for the alert email.")
            print("If you don't see it, check your spam/junk folder.")
        else:
            print("="*60)
            print("❌ FAILED! Email was not sent")
            print("="*60)
            print("\nCheck the error messages above for details.")
            
    except Exception as e:
        print("="*60)
        print(f"❌ ERROR: {str(e)}")
        print("="*60)
        print("\nCommon issues:")
        print("  1. Incorrect email/password")
        print("  2. Need to use App Password (for Gmail)")
        print("  3. SMTP server or port incorrect")
        print("  4. Firewall blocking connection")

if __name__ == "__main__":
    test_email_alert()
