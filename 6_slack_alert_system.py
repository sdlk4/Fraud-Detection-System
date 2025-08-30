import json
import requests
import logging
from datetime import datetime
import os
import sys

# Fix Windows Command Prompt Unicode issues
def safe_print(text):
    """Safely print text with Unicode character replacements for Windows compatibility"""
    replacements = {
        '🚨': '[ALERT]',
        '⚠️': '[WARNING]',
        '⚡': '[MEDIUM]',
        '🔒': '[FREEZE]',
        '📞': '[CALL]',
        '🔍': '[INVESTIGATE]',
        '📋': '[DOCUMENT]',
        '📊': '[VIEW]',
        '🤖': '[BOT]',
        '✓': '[OK]',
        '❌': '[ERROR]',
        '✅': '[SUCCESS]',
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

class SlackAlertSystem:
    """
    Slack alert system for fraud detection notifications using Slack SDK
    """
    
    def __init__(self, bot_token=None):
        """
        Initialize Slack alert system
        
        Args:
            bot_token: Slack bot token (starts with 'xoxb-')
        """
        self.bot_token = bot_token
        self.is_configured = False
        self.client = None
        
        # Try to import Slack SDK
        try:
            from slack_sdk import WebClient
            from slack_sdk.errors import SlackApiError
            
            if bot_token:
                self.client = WebClient(token=bot_token)
                self.is_configured = True
                logger.info("Slack client configured successfully")
        
        except ImportError:
            logger.warning("slack_sdk not installed. Install with: pip install slack_sdk")
            self.client = None
    
    def configure_token(self, bot_token):
        """
        Configure Slack bot token
        
        Args:
            bot_token: Slack bot token
        """
        try:
            from slack_sdk import WebClient
            self.bot_token = bot_token
            self.client = WebClient(token=bot_token)
            self.is_configured = True
            logger.info("Slack bot token configured")
        except ImportError:
            logger.error("slack_sdk not available")
    
    def create_fraud_alert_blocks(self, transaction_details):
        """
        Create Slack blocks for fraud alert message
        
        Args:
            transaction_details: Dictionary with transaction information
        
        Returns:
            List of Slack blocks
        """
        
        txn_id = transaction_details.get('transaction_id', 'Unknown')
        amount = transaction_details.get('amount', 0)
        probability = transaction_details.get('fraud_probability', 0)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine risk level and color - FIXED: Removed emojis
        if probability >= 0.9:
            risk_level = "CRITICAL"
            color = "#FF0000"  # Red
            emoji_text = "[CRITICAL]"
        elif probability >= 0.7:
            risk_level = "HIGH"
            color = "#FF6600"  # Orange
            emoji_text = "[WARNING]"
        else:
            risk_level = "MEDIUM"
            color = "#FFCC00"  # Yellow
            emoji_text = "[MEDIUM]"
        
        # Create Slack blocks - FIXED: Replaced emojis with text
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji_text} FRAUD DETECTION ALERT",
                    "emoji": False
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Suspicious transaction detected!*\nRisk Level: *{risk_level}* | Probability: *{probability:.1%}*"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Transaction ID:*\n{txn_id}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Amount:*\n${amount:,.2f}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Fraud Probability:*\n{probability:.1%}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Detection Time:*\n{timestamp}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Recommended Actions:*\n• [FREEZE] Freeze the account immediately\n• [CALL] Contact customer for verification\n• [INVESTIGATE] Investigate transaction patterns\n• [DOCUMENT] Document findings for compliance"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "[FREEZE] Freeze Account",
                            "emoji": False
                        },
                        "style": "danger",
                        "value": f"freeze_account_{txn_id}",
                        "action_id": "freeze_account"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "[CALL] Contact Customer",
                            "emoji": False
                        },
                        "style": "primary",
                        "value": f"contact_customer_{txn_id}",
                        "action_id": "contact_customer"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "[VIEW] View Details",
                            "emoji": False
                        },
                        "value": f"view_details_{txn_id}",
                        "action_id": "view_details"
                    }
                ]
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"[BOT] Generated by Fraud Detection System | {timestamp}"
                    }
                ]
            }
        ]
        
        return blocks
    
    def send_fraud_alert(self, transaction_details, channel_id, thread_ts=None):
        """
        Send fraud alert to Slack channel
        
        Args:
            transaction_details: Dictionary with transaction information
            channel_id: Slack channel ID (e.g., 'C1234567890' or '#fraud-alerts')
            thread_ts: Timestamp of parent message (for threading)
        
        Returns:
            Boolean indicating success/failure and response data
        """
        
        if not self.is_configured or not self.client:
            logger.error("Slack client not configured or slack_sdk not available")
            return False, None
        
        try:
            from slack_sdk.errors import SlackApiError
            
            # Create message blocks
            blocks = self.create_fraud_alert_blocks(transaction_details)
            
            # Prepare message text (fallback for notifications) - FIXED: Removed emoji
            txn_id = transaction_details.get('transaction_id', 'Unknown')
            amount = transaction_details.get('amount', 0)
            probability = transaction_details.get('fraud_probability', 0)
            
            fallback_text = f"[ALERT] FRAUD ALERT: Transaction {txn_id} (${amount:,.2f}) flagged with {probability:.1%} probability"
            
            # Send message to Slack
            response = self.client.chat_postMessage(
                channel=channel_id,
                text=fallback_text,
                blocks=blocks,
                thread_ts=thread_ts,
                unfurl_links=False,
                unfurl_media=False
            )
            
            if response["ok"]:
                logger.info(f"Fraud alert sent successfully to {channel_id}")
                self._log_slack_alert(transaction_details, channel_id, response["ts"])
                return True, response
            else:
                logger.error(f"Failed to send Slack message: {response}")
                return False, response
                
        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            return False, None
        except Exception as e:
            logger.error(f"Unexpected error sending Slack alert: {str(e)}")
            return False, None
    
    def send_thread_update(self, parent_ts, channel_id, update_message):
        """
        Send update message in thread
        
        Args:
            parent_ts: Timestamp of parent message
            channel_id: Slack channel ID
            update_message: Update message text
        
        Returns:
            Boolean indicating success/failure
        """
        
        if not self.is_configured or not self.client:
            return False
        
        try:
            response = self.client.chat_postMessage(
                channel=channel_id,
                text=update_message,
                thread_ts=parent_ts
            )
            return response["ok"]
        except Exception as e:
            logger.error(f"Error sending thread update: {str(e)}")
            return False
    
    def _log_slack_alert(self, transaction_details, channel_id, message_ts):
        """
        Log the sent Slack alert for record keeping
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'transaction_id': transaction_details.get('transaction_id'),
            'amount': transaction_details.get('amount'),
            'fraud_probability': transaction_details.get('fraud_probability'),
            'channel_id': channel_id,
            'message_ts': message_ts,
            'alert_sent': True
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Append to daily log file
        log_filename = f"logs/slack_alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
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
            logger.error(f"Failed to log Slack alert: {str(e)}")

# Alternative implementation using webhook (doesn't require slack_sdk)
class SlackWebhookAlertSystem:
    """
    Alternative Slack alert system using incoming webhooks
    """
    
    def __init__(self, webhook_url=None):
        """
        Initialize webhook-based Slack alert system
        
        Args:
            webhook_url: Slack incoming webhook URL
        """
        self.webhook_url = webhook_url
        self.is_configured = webhook_url is not None
    
    def configure_webhook(self, webhook_url):
        """
        Configure webhook URL
        
        Args:
            webhook_url: Slack incoming webhook URL
        """
        self.webhook_url = webhook_url
        self.is_configured = True
    
    def send_fraud_alert_webhook(self, transaction_details):
        """
        Send fraud alert using webhook
        
        Args:
            transaction_details: Dictionary with transaction information
        
        Returns:
            Boolean indicating success/failure
        """
        
        if not self.is_configured:
            logger.error("Webhook URL not configured")
            return False
        
        try:
            txn_id = transaction_details.get('transaction_id', 'Unknown')
            amount = transaction_details.get('amount', 0)
            probability = transaction_details.get('fraud_probability', 0)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create webhook payload - FIXED: Removed emoji
            payload = {
                "text": f"[ALERT] FRAUD ALERT: Transaction {txn_id}",
                "attachments": [
                    {
                        "color": "#FF0000" if probability >= 0.9 else "#FF6600",
                        "title": "[ALERT] Fraud Detection Alert",
                        "fields": [
                            {
                                "title": "Transaction ID",
                                "value": txn_id,
                                "short": True
                            },
                            {
                                "title": "Amount",
                                "value": f"${amount:,.2f}",
                                "short": True
                            },
                            {
                                "title": "Fraud Probability",
                                "value": f"{probability:.1%}",
                                "short": True
                            },
                            {
                                "title": "Detection Time",
                                "value": timestamp,
                                "short": True
                            }
                        ],
                        "footer": "Fraud Detection System",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            # Send webhook request
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("Fraud alert sent successfully via webhook")
                return True
            else:
                logger.error(f"Webhook request failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error sending webhook: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending webhook: {str(e)}")
            return False

def send_slack_fraud_alert(transaction_id, amount, fraud_probability, 
                          bot_token, channel_id):
    """
    Simplified function to send Slack fraud alert
    
    Args:
        transaction_id: Transaction identifier
        amount: Transaction amount
        fraud_probability: Probability of fraud (0-1)
        bot_token: Slack bot token
        channel_id: Slack channel ID
    
    Returns:
        Boolean indicating success/failure
    """
    
    # Create Slack alert system
    slack_system = SlackAlertSystem(bot_token)
    
    # Prepare transaction details
    transaction_details = {
        'transaction_id': transaction_id,
        'amount': amount,
        'fraud_probability': fraud_probability
    }
    
    # Send alert
    success, response = slack_system.send_fraud_alert(transaction_details, channel_id)
    return success

def test_slack_system():
    """
    Test the Slack alert system
    """
    safe_print("=== SLACK ALERT SYSTEM TEST ===\n")
    
    # Sample transaction details
    sample_transaction = {
        'transaction_id': 'TXN_789012',
        'amount': 3500.25,
        'fraud_probability': 0.91
    }
    
    safe_print("Sample Transaction Details:")
    safe_print(f"ID: {sample_transaction['transaction_id']}")
    safe_print(f"Amount: ${sample_transaction['amount']:.2f}")
    safe_print(f"Fraud Probability: {sample_transaction['fraud_probability']:.1%}")
    
    # FIXED: Replaced Unicode warning with safe_print
    safe_print("\n[WARNING] SLACK CONFIGURATION REQUIRED [WARNING]")
    safe_print("To test the Slack system, you need:")
    safe_print("1. Slack bot token (starts with 'xoxb-')")
    safe_print("2. Channel ID (e.g., 'C1234567890' or '#fraud-alerts')")
    safe_print("\nTo get these:")
    safe_print("• Create a Slack app at https://api.slack.com/apps")
    safe_print("• Add 'chat:write' and 'chat:write.public' scopes")
    safe_print("• Install the app to your workspace")
    safe_print("• Copy the Bot User OAuth Token")
    
    # Sample configuration (replace with actual values to test)
    bot_token = "xoxb-your-bot-token-here"
    channel_id = "#fraud-alerts"  # or channel ID like "C1234567890"
    webhook_url = "https://hooks.slack.com/services/your/webhook/url"
    
    safe_print(f"\nConfigured bot token: {bot_token[:15]}..." if len(bot_token) > 15 else bot_token)
    safe_print(f"Configured channel: {channel_id}")
    
    # Test with Slack SDK (if available)
    safe_print("\n=== TESTING SLACK SDK APPROACH ===")
    try:
        slack_system = SlackAlertSystem()
        if slack_system.client is None:
            safe_print("[WARNING] slack_sdk not installed. Install with: pip install slack_sdk")
        else:
            slack_system.configure_token(bot_token)
            blocks = slack_system.create_fraud_alert_blocks(sample_transaction)
            safe_print("[OK] Slack blocks created successfully")
            safe_print(f"Number of blocks: {len(blocks)}")
            
    except Exception as e:
        safe_print(f"[ERROR] Error testing Slack SDK: {str(e)}")
    
    # Test with webhook approach
    safe_print("\n=== TESTING WEBHOOK APPROACH ===")
    try:
        webhook_system = SlackWebhookAlertSystem(webhook_url)
        safe_print("[OK] Webhook system initialized")
        safe_print("[OK] Webhook payload structure ready")
        
    except Exception as e:
        safe_print(f"[ERROR] Error testing webhook: {str(e)}")
    
    safe_print("\n=== SLACK SYSTEM TEST COMPLETE ===")
    safe_print("\nTo actually send Slack messages:")
    safe_print("1. Replace sample tokens/URLs with real ones")
    safe_print("2. Uncomment the send alert calls below")
    safe_print("3. Run the script")

def main():
    """
    Main function demonstrating Slack alert system
    """
    
    safe_print("FRAUD DETECTION SLACK ALERT SYSTEM")
    safe_print("=" * 50)
    
    # Test the system
    test_slack_system()
    
    safe_print("\n" + "=" * 50)
    safe_print("USAGE EXAMPLES:")
    safe_print("=" * 50)
    
    safe_print("\n1. Simple usage with bot token:")
    safe_print("""
    success = send_slack_fraud_alert(
        transaction_id='TXN_789012',
        amount=3500.25,
        fraud_probability=0.91,
        bot_token='xoxb-your-bot-token',
        channel_id='#fraud-alerts'
    )
    """)
    
    safe_print("\n2. Advanced usage with SDK:")
    safe_print("""
    slack_system = SlackAlertSystem('xoxb-your-bot-token')
    
    transaction_details = {
        'transaction_id': 'TXN_789012',
        'amount': 3500.25,
        'fraud_probability': 0.91
    }
    
    success, response = slack_system.send_fraud_alert(
        transaction_details, 
        '#fraud-alerts'
    )
    """)
    
    safe_print("\n3. Webhook approach:")
    safe_print("""
    webhook_system = SlackWebhookAlertSystem(
        'https://hooks.slack.com/services/your/webhook/url'
    )
    
    success = webhook_system.send_fraud_alert_webhook(transaction_details)
    """)
    
    safe_print("\n" + "=" * 50)
    safe_print("SETUP INSTRUCTIONS:")
    safe_print("=" * 50)
    safe_print("1. Create a Slack app: https://api.slack.com/apps")
    safe_print("2. Add Bot Token Scopes: chat:write, chat:write.public")
    safe_print("3. Install app to workspace")
    safe_print("4. Copy Bot User OAuth Token (starts with 'xoxb-')")
    safe_print("5. Get channel ID from Slack (right-click channel → Copy link)")
    safe_print("6. Alternative: Set up incoming webhook for simpler integration")
    
    return None

if __name__ == "__main__":
    main()