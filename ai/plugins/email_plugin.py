"""
Gmail Integration Plugin for A.L.I.C.E
Handles email reading, searching, and sending
"""

import os
import logging
import pickle
from typing import Dict, List, Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import base64

logger = logging.getLogger(__name__)

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
          'https://www.googleapis.com/auth/gmail.send',
          'https://www.googleapis.com/auth/gmail.modify']


class GmailPlugin:
    """Gmail integration for reading and sending emails"""
    
    def __init__(self):
        self.service = None
        self.creds = None
        self.user_email = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Gmail API using OAuth2"""
        token_path = 'config/cred/gmail_token.pickle'
        creds_path = 'config/cred/gmail_credentials.json'
        
        # Load saved credentials
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                self.creds = pickle.load(token)
        
        # Refresh or get new credentials
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                    logger.info("[OK] Gmail credentials refreshed")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to refresh credentials: {e}")
                    self.creds = None
            
            if not self.creds:
                if not os.path.exists(creds_path):
                    logger.warning("[WARNING] Gmail credentials not found. Run setup first.")
                    logger.info("Get credentials from: https://console.cloud.google.com/apis/credentials")
                    return
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                    self.creds = flow.run_local_server(port=0)
                    logger.info("[OK] Gmail authentication successful")
                except Exception as e:
                    logger.error(f"[ERROR] Authentication failed: {e}")
                    return
            
            # Save credentials
            with open(token_path, 'wb') as token:
                pickle.dump(self.creds, token)
        
        # Build service
        try:
            self.service = build('gmail', 'v1', credentials=self.creds)
            # Get user email
            profile = self.service.users().getProfile(userId='me').execute()
            self.user_email = profile.get('emailAddress')
            logger.info(f"[OK] Gmail connected: {self.user_email}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to build Gmail service: {e}")
    
    def get_recent_emails(self, max_results: int = 10) -> List[Dict]:
        """Get recent emails from inbox"""
        if not self.service:
            return []
        
        try:
            results = self.service.users().messages().list(
                userId='me',
                labelIds=['INBOX'],
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for msg in messages:
                msg_data = self.service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='metadata',
                    metadataHeaders=['From', 'Subject', 'Date']
                ).execute()
                
                headers = msg_data.get('payload', {}).get('headers', [])
                email_info = {
                    'id': msg['id'],
                    'from': next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown'),
                    'subject': next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject'),
                    'date': next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown'),
                    'unread': 'UNREAD' in msg_data.get('labelIds', [])
                }
                emails.append(email_info)
            
            return emails
        except Exception as e:
            logger.error(f"[ERROR] Failed to get emails: {e}")
            return []
    
    def get_email_content(self, email_id: str) -> Optional[str]:
        """Get full content of an email"""
        if not self.service:
            return None
        
        try:
            msg = self.service.users().messages().get(
                userId='me',
                id=email_id,
                format='full'
            ).execute()
            
            payload = msg.get('payload', {})
            parts = payload.get('parts', [])
            
            # Try to get plain text content
            body = ''
            if parts:
                for part in parts:
                    if part.get('mimeType') == 'text/plain':
                        data = part.get('body', {}).get('data', '')
                        if data:
                            body = base64.urlsafe_b64decode(data).decode('utf-8')
                            break
            else:
                # No parts, try body directly
                data = payload.get('body', {}).get('data', '')
                if data:
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
            
            return body
        except Exception as e:
            logger.error(f"[ERROR] Failed to get email content: {e}")
            return None
    
    def search_emails(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search emails by query"""
        if not self.service:
            return []
        
        try:
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for msg in messages:
                msg_data = self.service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='metadata',
                    metadataHeaders=['From', 'Subject', 'Date']
                ).execute()
                
                headers = msg_data.get('payload', {}).get('headers', [])
                email_info = {
                    'id': msg['id'],
                    'from': next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown'),
                    'subject': next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject'),
                    'date': next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
                }
                emails.append(email_info)
            
            return emails
        except Exception as e:
            logger.error(f"[ERROR] Failed to search emails: {e}")
            return []
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send an email"""
        if not self.service:
            return False
        
        try:
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            self.service.users().messages().send(
                userId='me',
                body={'raw': raw}
            ).execute()
            
            logger.info(f"[OK] Email sent to {to}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to send email: {e}")
            return False
    
    def mark_as_read(self, email_id: str) -> bool:
        """Mark email as read"""
        if not self.service:
            return False
        
        try:
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to mark as read: {e}")
            return False
    
    def mark_as_unread(self, email_id: str) -> bool:
        """Mark email as unread"""
        if not self.service:
            return False
        
        try:
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'addLabelIds': ['UNREAD']}
            ).execute()
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to mark as unread: {e}")
            return False
    
    def archive_email(self, email_id: str) -> bool:
        """Archive an email (remove from inbox)"""
        if not self.service:
            return False
        
        try:
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['INBOX']}
            ).execute()
            logger.info(f"[OK] Email archived")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to archive: {e}")
            return False
    
    def delete_email(self, email_id: str) -> bool:
        """Move email to trash"""
        if not self.service:
            return False
        
        try:
            self.service.users().messages().trash(
                userId='me',
                id=email_id
            ).execute()
            logger.info(f"[OK] Email moved to trash")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to delete: {e}")
            return False
    
    def star_email(self, email_id: str) -> bool:
        """Star/flag an email"""
        if not self.service:
            return False
        
        try:
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'addLabelIds': ['STARRED']}
            ).execute()
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to star: {e}")
            return False
    
    def unstar_email(self, email_id: str) -> bool:
        """Remove star from email"""
        if not self.service:
            return False
        
        try:
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['STARRED']}
            ).execute()
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to unstar: {e}")
            return False
    
    def get_unread_count(self) -> int:
        """Get count of unread emails"""
        if not self.service:
            return 0
        
        try:
            results = self.service.users().messages().list(
                userId='me',
                labelIds=['INBOX', 'UNREAD']
            ).execute()
            return results.get('resultSizeEstimate', 0)
        except Exception as e:
            logger.error(f"[ERROR] Failed to get unread count: {e}")
            return 0
    
    def get_emails_by_sender(self, sender: str, max_results: int = 10) -> List[Dict]:
        """Get emails from a specific sender"""
        query = f"from:{sender}"
        return self.search_emails(query, max_results)
    
    def get_emails_with_attachments(self, max_results: int = 10) -> List[Dict]:
        """Get emails that have attachments"""
        query = "has:attachment"
        return self.search_emails(query, max_results)
    
    def reply_to_email(self, email_id: str, reply_body: str) -> bool:
        """Reply to an email"""
        if not self.service:
            return False
        
        try:
            # Get original message
            original = self.service.users().messages().get(
                userId='me',
                id=email_id,
                format='metadata',
                metadataHeaders=['From', 'Subject', 'Message-ID']
            ).execute()
            
            headers = original.get('payload', {}).get('headers', [])
            original_from = next((h['value'] for h in headers if h['name'] == 'From'), '')
            original_subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
            message_id = next((h['value'] for h in headers if h['name'] == 'Message-ID'), '')
            
            # Extract email from "Name <email>" format
            import re
            email_match = re.search(r'<(.+?)>', original_from)
            to_email = email_match.group(1) if email_match else original_from
            
            # Create reply
            subject = f"Re: {original_subject}" if not original_subject.startswith('Re:') else original_subject
            
            message = MIMEText(reply_body)
            message['to'] = to_email
            message['subject'] = subject
            message['In-Reply-To'] = message_id
            message['References'] = message_id
            
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            self.service.users().messages().send(
                userId='me',
                body={'raw': raw, 'threadId': original.get('threadId')}
            ).execute()
            
            logger.info(f"[OK] Reply sent to {to_email}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to reply: {e}")
            return False
