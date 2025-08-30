import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import os
import json
import logging
import sys

# Fix Windows Command Prompt Unicode issues
def safe_print(text):
    """Safely print text with Unicode character replacements for Windows compatibility"""
    replacements = {
        '🚨': '[ALERT]',
        '📧': '[EMAIL]',
        '🔒': '[SECURE]',
        '📞': '[CALL]',
        '🔍': '[INVESTIGATE]',
        '📊': '[REVIEW]',
        '📋': '[DOCUMENT]',
        '⚠️': '[WARNING]',
        '✅': '[SUCCESS]',
        '❌': '[ERROR]',
        '⛔': '[STOP]',
    }
    
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
    
    try:
        print(text)
    except UnicodeEncodeError:
        # Final fallback - convert to ASCII
        ascii_text = text.encode('ascii', 'ignore').decode('ascii')
        print(ascii_text)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailAlertSystem:
    """
    Email alert system for fraud detection notifications
    """
    
    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=587):
        """
        Initialize email alert system
        
        Args:
            smtp_server: SMTP server address (default: Gmail)
            smtp_port: SMTP port number (default: 587 for TLS)
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = None
        self.sender_password = None
        self.is_configured = False
    
    def configure_credentials(self, sender_email, sender_password):
        """
        Configure email credentials
        
        Args:
            sender_email: Sender's email address
            sender_password: App password (not regular password for Gmail)
        """
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.is_configured = True
        logger.info(f"Email system configured for: {sender_email}")
    
    def create_fraud_alert_email(self, transaction_details, recipient_email):
        """
        Create fraud alert email message
        
        Args:
            transaction_details: Dictionary with transaction information
            recipient_email: Recipient's email address
        
        Returns:
            MIMEMultipart email message
        """
        
        # Extract transaction details
        txn_id = transaction_details.get('transaction_id', 'Unknown')
        amount = transaction_details.get('amount', 'Unknown')
        probability = transaction_details.get('fraud_probability', 0)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create message container - FIXED: Removed Unicode emoji
        message = MIMEMultipart("alternative")
        message["Subject"] = f"[FRAUD ALERT] - Transaction {txn_id}"
        message["From"] = self.sender_email
        message["To"] = recipient_email
        message["X-Priority"] = "1"  # High priority
        
        # Create HTML content - FIXED: Replaced emojis with text
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .alert-container {{
                    max-width: 600px;
                    margin: 0 auto;
                    font-family: Arial, sans-serif;
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                }}
                .alert-header {{
                    background-color: #dc3545;
                    color: white;
                    text-align: center;
                    padding: 20px;
                    border-radius: 10px 10px 0 0;
                    margin: -20px -20px 20px -20px;
                }}
                .alert-content {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .transaction-details {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-left: 4px solid #dc3545;
                    margin: 15px 0;
                }}
                .detail-row {{
                    margin: 10px 0;
                    padding: 5px 0;
                    border-bottom: 1px solid #eee;
                }}
                .label {{
                    font-weight: bold;
                    color: #333;
                    display: inline-block;
                    width: 150px;
                }}
                .value {{
                    color: #666;
                }}
                .high-risk {{
                    color: #dc3545;
                    font-weight: bold;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    color: #666;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="alert-container">
                <div class="alert-header">
                    <h1>[ALERT] FRAUD DETECTION ALERT [ALERT]</h1>
                    <p>Suspicious transaction detected</p>
                </div>
                
                <div class="alert-content">
                    <h2>Transaction Details</h2>
                    <div class="transaction-details">
                        <div class="detail-row">
                            <span class="label">Transaction ID:</span>
                            <span class="value">{txn_id}</span>
                        </div>
                        <div class="detail-row">
                            <span class="label">Amount:</span>
                            <span class="value high-risk">${amount:.2f}</span>
                        </div>
                        <div class="detail-row">
                            <span class="label">Fraud Probability:</span>
                            <span class="value high-risk">{probability:.2%}</span>
                        </div>
                        <div class="detail-row">
                            <span class="label">Detection Time:</span>
                            <span class="value">{timestamp}</span>
                        </div>
                        <div class="detail-row">
                            <span class="label">Risk Level:</span>
                            <span class="value high-risk">HIGH RISK</span>
                        </div>
                    </div>
                    
                    <h3>Recommended Actions:</h3>
                    <ul>
                        <li>[SECURE] Immediately freeze the associated account</li>
                        <li>[CALL] Contact the customer to verify the transaction</li>
                        <li>[INVESTIGATE] Investigate transaction patterns</li>
                        <li>[REVIEW] Review related transactions from the same account</li>
                        <li>[DOCUMENT] Document all findings for compliance</li>
                    </ul>
                    
                    <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin-top: 15px;">
                        <strong>[WARNING] Important:</strong> This alert was generated automatically by the fraud detection system. 
                        Please follow your organization's fraud response procedures immediately.
                    </div>
                </div>
                
                <div class="footer">
                    <p>This is an automated alert from the Fraud Detection System</p>
                    <p>Generated on {timestamp}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version - FIXED: Replaced emojis with text
        text_content = f"""
        [ALERT] FRAUD DETECTION ALERT [ALERT]
        
        A suspicious transaction has been detected by our fraud detection system.
        
        TRANSACTION DETAILS:
        ==================
        Transaction ID: {txn_id}
        Amount: ${amount:.2f}
        Fraud Probability: {probability:.2%}
        Detection Time: {timestamp}
        Risk Level: HIGH RISK
        
        RECOMMENDED ACTIONS:
        ===================
        1. Immediately freeze the associated account
        2. Contact the customer to verify the transaction
        3. Investigate transaction patterns
        4. Review related transactions from the same account
        5. Document all findings for compliance
        
        [WARNING] IMPORTANT: This alert was generated automatically by the fraud detection system.
        Please follow your organization's fraud response procedures immediately.
        
        ---
        This is an automated alert from the Fraud Detection System
        Generated on {timestamp}
        """
        
        # Create MIMEText objects
        text_part = MIMEText(text_content, "plain")
        html_part = MIMEText(html_content, "html")
        
        # Add parts to message
        message.attach(text_part)
        message.attach(html_part)
        
        return message
    
    def send_fraud_alert(self, transaction_details, recipient_email, cc_emails=None):
        """
        Send fraud alert email
        
        Args:
            transaction_details: Dictionary with transaction information
            recipient_email: Primary recipient's email address
            cc_emails: List of CC email addresses (optional)
        
        Returns:
            Boolean indicating success/failure
        """
        
        if not self.is_configured:
            logger.error("Email system not configured. Please set credentials first.")
            return False
        
        try:
            # Create email message
            message = self.create_fraud_alert_email(transaction_details, recipient_email)
            
            # Add CC recipients if provided
            if cc_emails:
                message["Cc"] = ", ".join(cc_emails)
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                
                # Prepare recipient list
                recipients = [recipient_email]
                if cc_emails:
                    recipients.extend(cc_emails)
                
                # Send email
                text = message.as_string()
                server.sendmail(self.sender_email, recipients, text)
            
            logger.info(f"Fraud alert sent successfully to {recipient_email}")
            
            # Log the alert for record keeping
            self._log_alert(transaction_details, recipient_email)
            
            return True
            
        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP Authentication failed. Check your credentials.")
            return False
        except smtplib.SMTPRecipientsRefused:
            logger.error(f"Recipient email address rejected: {recipient_email}")
            return False
        except smtplib.SMTPServerDisconnected:
            logger.error("SMTP server disconnected unexpectedly.")
            return False
        except Exception as e:
            logger.error(f"Failed to send fraud alert: {str(e)}")
            return False
    
    def _log_alert(self, transaction_details, recipient_email):
        """
        Log the sent alert for record keeping
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'transaction_id': transaction_details.get('transaction_id'),
            'amount': transaction_details.get('amount'),
            'fraud_probability': transaction_details.get('fraud_probability'),
            'recipient': recipient_email,
            'alert_sent': True
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Append to daily log file
        log_filename = f"logs/fraud_alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            # Read existing logs
            if os.path.exists(log_filename):
                with open(log_filename, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Add new log entry
            logs.append(log_entry)
            
            # Write back to file
            with open(log_filename, 'w') as f:
                json.dump(logs, f, indent=4)
                
        except Exception as e:
            logger.error(f"Failed to log alert: {str(e)}")

def send_fraud_email_alert(transaction_id, amount, fraud_probability, 
                          sender_email, sender_password, recipient_email,
                          cc_emails=None):
    """
    Simplified function to send fraud email alert
    
    Args:
        transaction_id: Transaction identifier
        amount: Transaction amount
        fraud_probability: Probability of fraud (0-1)
        sender_email: Sender's email address
        sender_password: App password for sender's email
        recipient_email: Recipient's email address
        cc_emails: List of CC email addresses (optional)
    
    Returns:
        Boolean indicating success/failure
    """
    
    # Create email alert system
    email_system = EmailAlertSystem()
    email_system.configure_credentials(sender_email, sender_password)
    
    # Prepare transaction details
    transaction_details = {
        'transaction_id': transaction_id,
        'amount': amount,
        'fraud_probability': fraud_probability
    }
    
    # Send alert
    return email_system.send_fraud_alert(transaction_details, recipient_email, cc_emails)

def test_email_system():
    """
    Test the email alert system with sample data
    """
    safe_print("=== EMAIL ALERT SYSTEM TEST ===\n")
    
    # Sample transaction details
    sample_transaction = {
        'transaction_id': 'TXN_123456',
        'amount': 2500.75,
        'fraud_probability': 0.87
    }
    
    safe_print("Sample Transaction Details:")
    safe_print(f"ID: {sample_transaction['transaction_id']}")
    safe_print(f"Amount: ${sample_transaction['amount']:.2f}")
    safe_print(f"Fraud Probability: {sample_transaction['fraud_probability']:.2%}")
    
    # Email configuration
    safe_print("\n[WARNING] EMAIL CONFIGURATION REQUIRED [WARNING]")
    safe_print("To test the email system, you need to provide:")
    safe_print("1. Sender email address (Gmail account)")
    safe_print("2. App password (not regular password)")
    safe_print("3. Recipient email address")
    safe_print("\nFor Gmail app passwords, visit: https://support.google.com/accounts/answer/185833")
    
    # Simulate email configuration
    sender_email = "your_email@gmail.com"
    sender_password = "your_app_password"
    recipient_email = "fraud_team@company.com"
    cc_emails = ["security@company.com", "manager@company.com"]
    
    safe_print(f"\nConfigured sender: {sender_email}")
    safe_print(f"Configured recipient: {recipient_email}")
    safe_print(f"CC recipients: {', '.join(cc_emails) if cc_emails else 'None'}")
    
    # Create and configure email system
    email_system = EmailAlertSystem()
    
    # Test email creation
    safe_print("\n=== TESTING EMAIL CREATION ===")
    try:
        email_system.configure_credentials(sender_email, sender_password)
        message = email_system.create_fraud_alert_email(sample_transaction, recipient_email)
        safe_print("Email message created successfully")
        
        # FIXED: Safely print the subject line
        subject = message['Subject']
        safe_print(f"Subject: {subject}")
        safe_print(f"From: {message['From']}")
        safe_print(f"To: {message['To']}")
        
        # Preview email content
        email_content = message.as_string()
        safe_print(f"\nEmail preview (first 200 chars):")
        safe_print(email_content[:200] + "...")
        
    except Exception as e:
        safe_print(f"[ERROR] Error creating email: {str(e)}")
    
    safe_print("\n=== EMAIL SYSTEM TEST COMPLETE ===")
    safe_print("\nTo actually send emails:")
    safe_print("1. Replace the sample credentials with real ones")
    safe_print("2. Uncomment the send_fraud_alert() call below")
    safe_print("3. Run the script")
    
    return email_system

# Integration function for use with fraud detection pipeline
def integrate_with_fraud_detection(model_prediction_result, 
                                 email_config, 
                                 fraud_threshold=0.7):
    """
    Integration function to automatically send email alerts based on fraud detection results
    
    Args:
        model_prediction_result: Result from fraud detection model
        email_config: Dictionary with email configuration
        fraud_threshold: Threshold for sending alerts
    
    Returns:
        Boolean indicating if alert was sent
    """
    
    # Check if fraud detected above threshold
    if model_prediction_result.get('fraud_probability', 0) > fraud_threshold:
        
        logger.info(f"Fraud detected above threshold ({fraud_threshold}). Sending email alert...")
        
        # Send email alert
        success = send_fraud_email_alert(
            transaction_id=model_prediction_result.get('transaction_id'),
            amount=model_prediction_result.get('amount'),
            fraud_probability=model_prediction_result.get('fraud_probability'),
            sender_email=email_config['sender_email'],
            sender_password=email_config['sender_password'],
            recipient_email=email_config['recipient_email'],
            cc_emails=email_config.get('cc_emails')
        )
        
        if success:
            logger.info("[SUCCESS] Fraud alert email sent successfully")
        else:
            logger.error("[ERROR] Failed to send fraud alert email")
        
        return success
    
    else:
        logger.info(f"Fraud probability ({model_prediction_result.get('fraud_probability', 0):.4f}) below threshold ({fraud_threshold}). No email sent.")
        return False

def main():
    """
    Main function demonstrating email alert system
    """
    
    safe_print("FRAUD DETECTION EMAIL ALERT SYSTEM")
    safe_print("=" * 50)
    
    # Test the email system
    email_system = test_email_system()
    
    safe_print("\n" + "=" * 50)
    safe_print("USAGE EXAMPLES:")
    safe_print("=" * 50)
    
    safe_print("\n1. Simple usage:")
    safe_print("""
    success = send_fraud_email_alert(
        transaction_id='TXN_123456',
        amount=2500.75,
        fraud_probability=0.87,
        sender_email='your_email@gmail.com',
        sender_password='your_app_password',
        recipient_email='fraud_team@company.com'
    )
    """)
    
    safe_print("\n2. Integration with fraud detection:")
    safe_print("""
    email_config = {
        'sender_email': 'alerts@yourcompany.com',
        'sender_password': 'your_app_password',
        'recipient_email': 'fraud_team@yourcompany.com',
        'cc_emails': ['security@yourcompany.com', 'manager@yourcompany.com']
    }
    
    # After getting prediction from model
    prediction_result = {
        'transaction_id': 'TXN_789012',
        'amount': 3200.50,
        'fraud_probability': 0.92
    }
    
    alert_sent = integrate_with_fraud_detection(
        prediction_result, 
        email_config, 
        fraud_threshold=0.7
    )
    """)
    
    safe_print("\n" + "=" * 50)
    safe_print("SECURITY NOTES:")
    safe_print("=" * 50)
    safe_print("• Use app passwords, not regular passwords for Gmail")
    safe_print("• Store credentials securely (environment variables/config files)")
    safe_print("• Use SSL/TLS encryption for email transmission")
    safe_print("• Log all fraud alerts for compliance and auditing")
    safe_print("• Consider rate limiting to avoid spam filters")
    safe_print("• Test with internal emails first")
    
    return email_system

if __name__ == "__main__":
    email_system = main()