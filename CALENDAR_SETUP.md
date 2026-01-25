# Google Calendar Plugin Setup Instructions

The Calendar Plugin requires Google Calendar API access. Follow these steps to set it up:

## 1. Install Required Dependencies

```bash
pip install google-auth google-auth-oauthlib google-api-python-client
```

## 2. Set Up Google Calendar API

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Calendar API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Calendar API"
   - Click on it and press "Enable"

## 3. Create Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Choose "Desktop application" as the application type
4. Give it a name (e.g., "A.L.I.C.E Calendar")
5. Download the credentials JSON file

## 4. Install Credentials

1. Create a `cred` folder in your A.L.I.C.E directory if it doesn't exist
2. Rename the downloaded file to `calendar_credentials.json`
3. Place it in the `cred/` folder

## 5. First-Time Setup

1. Restart A.L.I.C.E
2. Try a calendar command like "show my calendar"
3. A browser will open for OAuth authentication
4. Grant permission to access your calendar
5. The plugin will save your authentication token for future use

## Supported Commands

### View Calendar
- "Show my calendar"
- "What's on my calendar today?"
- "Show my schedule for tomorrow"
- "What meetings do I have this week?"

### Create Events
- "Schedule a meeting with John tomorrow at 2pm"
- "Create an appointment for 3pm today"
- "Book a call for next Monday at 10am"
- "Set up a meeting about project review"

### Search Events
- "Find my meeting with Sarah"
- "Look for project meetings"
- "Search for doctor appointments"

### Natural Language Support
- Times: "2pm", "10:30am", "morning", "afternoon", "evening"
- Dates: "today", "tomorrow", "next Monday", "this Friday"
- Duration: Automatically sets 1-hour meetings by default
- Locations: "at the office", "in conference room A", "via Zoom"

## Troubleshooting

### "Calendar is not set up yet" Error
- Make sure `calendar_credentials.json` is in the `cred/` folder
- Restart A.L.I.C.E after adding credentials

### Authentication Issues
- Delete `cred/calendar_token.pickle` and try again
- Check that Google Calendar API is enabled in your Google Cloud project
- Verify your credentials file is valid JSON

### Permission Errors
- Make sure you granted calendar access during OAuth flow
- Check that your Google account has access to the calendar you're trying to use

## Privacy & Security

- Your calendar data stays on your local machine and Google's servers
- A.L.I.C.E only requests read/write access to your calendar events
- Authentication tokens are stored locally in encrypted format
- No calendar data is shared with third parties

## Advanced Configuration

You can customize the calendar plugin by modifying `ai/calendar_plugin.py`:
- Change default timezone in `self.user_timezone`
- Modify default reminder time in `reminder_minutes`
- Add support for additional calendar providers