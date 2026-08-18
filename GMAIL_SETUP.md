# Gmail Setup Guide

To enable automatic email checking from your Gmail inbox, follow these steps:

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project:
   - Click on the project dropdown at the top
   - Click "NEW PROJECT"
   - Enter a name (e.g., "Email Spam Detector")
   - Click "CREATE"

## Step 2: Enable Gmail API

1. In the Cloud Console, search for "Gmail API"
2. Click on "Gmail API"
3. Click "ENABLE"

## Step 3: Create OAuth 2.0 Credentials

1. Go to "Credentials" in the left sidebar
2. Click "CREATE CREDENTIALS"
3. Select "OAuth client ID"
4. If prompted, configure the OAuth consent screen:
   - Select "External" for User Type
   - Fill in the application name and your email
   - Add scopes: `https://www.googleapis.com/auth/gmail.readonly`
   - Complete the consent screen setup
5. After consent screen is set up, go back to "CREATE CREDENTIALS"
6. Select "Desktop application" as Application Type
7. Click "CREATE"
8. Download the JSON file and save it in your project folder

## Step 4: Configure the App

1. Rename the downloaded JSON file to `credentials.json`
2. Place `credentials.json` in the same folder as `app.py`
3. The app will automatically create `gmail_token.json` on first login

## Step 5: Run the App

```bash
python -m streamlit run app.py
```

Navigate to the "📬 Gmail Inbox" tab and click "🔗 Connect to Gmail" to authenticate.

## Troubleshooting

- **File not found**: Make sure `credentials.json` is in the project directory
- **Authentication issues**: Delete `gmail_token.json` and try connecting again
- **API not enabled**: Check that Gmail API is enabled in your Google Cloud project

## Privacy

- Your email credentials are stored locally in `gmail_token.json`
- The app only reads your emails (readonly access)
- Emails are processed locally on your machine
