import streamlit as st
import pickle
import re
import os
import base64
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SESSION STATE
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "gmail_creds" not in st.session_state:
    st.session_state.gmail_creds = None

if "manual_result" not in st.session_state:
    st.session_state.manual_result = None

if "manual_sender" not in st.session_state:
    st.session_state.manual_sender = ""

if "manual_subject" not in st.session_state:
    st.session_state.manual_subject = ""

if "manual_body" not in st.session_state:
    st.session_state.manual_body = ""

if "classified_emails" not in st.session_state:
    st.session_state.classified_emails = None


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:

    with open(
        os.path.join(BASE_DIR, "spam_model.pkl"),
        "rb"
    ) as file:

        model = pickle.load(file)

    with open(
        os.path.join(BASE_DIR, "vectorizer.pkl"),
        "rb"
    ) as file:

        vectorizer = pickle.load(file)


except FileNotFoundError:

    st.error(
        """
        ❌ Model files not found.

        Make sure these files are in the same folder as app.py:

        • spam_model.pkl
        • vectorizer.pkl
        """
    )

    st.stop()


# ============================================================
# LOAD GRAPH FILES
# ============================================================

ACCURACY_GRAPH = os.path.join(BASE_DIR, "training_validation_accuracy.png")
LOSS_GRAPH = os.path.join(BASE_DIR, "training_validation_loss.png")


# ============================================================
# GMAIL CONFIGURATION
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly"
]

TOKEN_FILE = os.getenv(
    "GMAIL_TOKEN_PATH",
    os.path.join(BASE_DIR, "gmail_token.json")
)
CREDENTIALS_FILE = os.getenv(
    "GMAIL_CREDENTIALS_PATH",
    os.path.join(BASE_DIR, "credentials.json")
)

# Render/cloud deployments do not have a browser on the server.
# A previously authorized Gmail token should be supplied through
# the GMAIL_TOKEN_JSON environment variable.
IS_RENDER = bool(os.getenv("RENDER"))


# ============================================================
# PROFESSIONAL CSS
# ============================================================

if st.session_state.dark_mode:

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #0f172a;
        }

        .main .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            color: #f8fafc !important;
        }

        h2 {
            font-size: 1.9rem !important;
            font-weight: 650 !important;
            color: #f8fafc !important;
        }

        h3 {
            font-size: 1.5rem !important;
            font-weight: 650 !important;
            color: #f8fafc !important;
        }

        h4 {
            font-size: 1.25rem !important;
            color: #f8fafc !important;
        }

        p {
            color: #cbd5e1 !important;
            font-size: 25px !important;
            line-height: 1.6 !important;
        }

        label {
            color: #cbd5e1 !important;
            font-size: 16px !important;
            font-weight: 500 !important;
        }

        div[data-baseweb="input"] {
            background-color: #1e293b !important;
            border-color: #475569 !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="input"] input {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            font-size: 25px !important;
        }

        div[data-baseweb="textarea"] {
            background-color: #1e293b !important;
            border-color: #475569 !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="textarea"] textarea {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            font-size: 25px !important;
            line-height: 1.6 !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #94a3b8 !important;
            font-size: 25px !important;
        }

        .stButton button {
            font-size: 25px !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            min-height: 44px !important;
            padding: 8px 18px !important;
        }

        div[data-testid="stMetric"] {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 20px;
        }

        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 16px !important;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-size: 30px !important;
            font-weight: 700 !important;
        }

        div[data-testid="stExpander"] {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
        }

        button[data-baseweb="tab"] {
            color: #94a3b8 !important;
            font-size: 16px !important;
            font-weight: 500 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #60a5fa !important;
            font-weight: 600 !important;
        }

        div[data-testid="stCaptionContainer"] {
            font-size: 15px !important;
        }

        div[data-baseweb="select"] {
            font-size: 25px !important;
        }

        div[data-testid="stAlert"] {
            font-size: 25px !important;
        }

        hr {
            border-color: #334155 !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #f8fafc;
        }

        .main .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            color: #0f172a !important;
        }

        h2 {
            font-size: 1.9rem !important;
            font-weight: 650 !important;
            color: #0f172a !important;
        }

        h3 {
            font-size: 1.5rem !important;
            font-weight: 650 !important;
            color: #0f172a !important;
        }

        h4 {
            font-size: 1.25rem !important;
            color: #0f172a !important;
        }

        p {
            color: #334155 !important;
            font-size: 25px !important;
            line-height: 1.6 !important;
        }

        label {
            color: #334155 !important;
            font-size: 16px !important;
            font-weight: 500 !important;
        }

        div[data-baseweb="input"] {
            background-color: white !important;
            border-color: #cbd5e1 !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="input"] input {
            background-color: white !important;
            color: #0f172a !important;
            font-size: 25px !important;
        }

        div[data-baseweb="textarea"] {
            background-color: white !important;
            border-color: #cbd5e1 !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="textarea"] textarea {
            background-color: white !important;
            color: #0f172a !important;
            font-size: 25px !important;
            line-height: 1.6 !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #94a3b8 !important;
            font-size: 25px !important;
        }

        .stButton button {
            font-size: 25px !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            min-height: 44px !important;
            padding: 8px 18px !important;
        }

        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 20px;
        }

        div[data-testid="stMetricLabel"] {
            color: #334155 !important;
            font-size: 16px !important;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-size: 30px !important;
            font-weight: 700 !important;
        }

        div[data-testid="stExpander"] {
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
        }

        button[data-baseweb="tab"] {
            color: #64748b !important;
            font-size: 16px !important;
            font-weight: 500 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #2563eb !important;
            font-weight: 600 !important;
        }

        div[data-testid="stCaptionContainer"] {
            font-size: 15px !important;
        }

        div[data-baseweb="select"] {
            font-size: 25px !important;
        }

        div[data-testid="stAlert"] {
            font-size: 25px !important;
        }

        hr {
            border-color: #e2e8f0 !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# GMAIL AUTHENTICATION
# ============================================================

def _load_json_from_streamlit_secrets(key):
    """Disabled on Render; environment variables are used instead."""
    return None

def _load_gmail_credentials_config():
    """Load Google OAuth client configuration safely."""
    # 1. Render/environment variable
    raw = os.getenv("GMAIL_CREDENTIALS_JSON")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            st.error("❌ GMAIL_CREDENTIALS_JSON contains invalid JSON.")
            return None

    # 2. Streamlit secrets (local/cloud)
    secret_config = _load_json_from_streamlit_secrets("gmail_credentials")
    if secret_config:
        return secret_config

    # 3. Local credentials.json
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            st.error(f"❌ Could not read credentials.json: {e}")
            return None

    return None


def _load_gmail_token():
    """Load an authorized Gmail token from environment or local disk."""
    # Render/environment variable
    raw = os.getenv("GMAIL_TOKEN_JSON")
    if raw:
        try:
            return Credentials.from_authorized_user_info(
                json.loads(raw),
                SCOPES
            )
        except Exception as e:
            st.error(f"❌ GMAIL_TOKEN_JSON is invalid: {e}")
            return None

    # Local token file
    if os.path.exists(TOKEN_FILE):
        try:
            return Credentials.from_authorized_user_file(
                TOKEN_FILE,
                SCOPES
            )
        except Exception:
            return None

    return None


def _save_gmail_token(creds):
    """Save a token locally when possible."""
    try:
        with open(TOKEN_FILE, "w", encoding="utf-8") as token:
            token.write(creds.to_json())
    except Exception:
        # Render's filesystem should not be relied upon for credentials.
        pass


def authenticate_gmail():
    """Authenticate Gmail.

    Local:
        Uses credentials.json/Streamlit secrets and opens Google's OAuth
        browser flow.

    Render:
        Uses GMAIL_TOKEN_JSON. A browser-based OAuth flow cannot be opened
        from the Render server, so the token must be authorized beforehand.
    """
    creds = _load_gmail_token()

    # Existing valid token
    if creds and creds.valid:
        return creds

    # Refresh expired token
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_gmail_token(creds)
            return creds
        except Exception:
            creds = None

    # Render cannot use run_local_server() because there is no local
    # browser/callback endpoint available to the user.
    if IS_RENDER:
        st.error(
            """
            ❌ Gmail is not authorized on the Render server.

            Add these Render Environment Variables:

            • GMAIL_CREDENTIALS_JSON — complete credentials.json content
            • GMAIL_TOKEN_JSON — complete authorized gmail_token.json content

            Do not upload credentials.json or gmail_token.json to GitHub.
            """
        )
        return None

    # Local authentication
    google_credentials = _load_gmail_credentials_config()

    if not google_credentials:
        st.error(
            """
            ❌ Gmail OAuth credentials were not found.

            For local development, place credentials.json next to app.py
            or configure the gmail_credentials Streamlit secret.
            """
        )
        return None

    try:
        flow = InstalledAppFlow.from_client_config(
            google_credentials,
            SCOPES
        )

        creds = flow.run_local_server(port=0)

        _save_gmail_token(creds)

        return creds

    except Exception as e:
        st.error(f"❌ Gmail authentication failed: {e}")
        return None


# ============================================================
# EXTRACT EMAIL BODY
# ============================================================

def extract_email_body(payload):

    if not payload:
        return ""

    mime_type = payload.get(
        "mimeType",
        ""
    )

    if mime_type == "text/plain":

        data = payload.get(
            "body",
            {}
        ).get(
            "data",
            ""
        )

        if data:

            try:

                return base64.urlsafe_b64decode(
                    data
                ).decode(
                    "utf-8",
                    errors="ignore"
                )

            except Exception:

                return ""

    for part in payload.get(
        "parts",
        []
    ):

        result = extract_email_body(
            part
        )

        if result:
            return result

    return ""


# ============================================================
# FETCH GMAIL EMAILS
# ============================================================

def fetch_emails(creds):

    try:

        service = build(
            "gmail",
            "v1",
            credentials=creds
        )

        results = service.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            maxResults=100
        ).execute()

        messages = results.get(
            "messages",
            []
        )

        emails = []

        for message in messages:

            email_data = service.users().messages().get(
                userId="me",
                id=message["id"],
                format="full"
            ).execute()

            payload = email_data.get(
                "payload",
                {}
            )

            headers = payload.get(
                "headers",
                []
            )

            subject = next(
                (
                    h["value"]
                    for h in headers
                    if h["name"].lower() == "subject"
                ),
                "No Subject"
            )

            sender = next(
                (
                    h["value"]
                    for h in headers
                    if h["name"].lower() == "from"
                ),
                "Unknown Sender"
            )

            body = extract_email_body(
                payload
            )

            emails.append(
                {
                    "id": message["id"],
                    "subject": subject,
                    "sender": sender,
                    "body": body
                }
            )

        return emails

    except Exception as e:

        st.error(
            f"❌ Error fetching emails: {e}"
        )

        return []


# ============================================================
# SPAM KEYWORDS
# ============================================================

def extract_keywords(text):

    spam_keywords = {

        "free",
        "winner",
        "won",
        "prize",
        "offer",
        "cash",
        "money",
        "urgent",
        "bonus",
        "click",
        "claim",
        "congratulations",
        "discount",
        "limited",
        "reward",
        "lottery",
        "investment",
        "guaranteed",
        "promotion",
        "selected",
        "verify",
        "account",
        "password"
    }

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text.lower()
    )

    found = [
        word
        for word in words
        if word in spam_keywords
    ]

    return sorted(
        list(set(found))
    )


# ============================================================
# CLASSIFY EMAIL
# ============================================================

def classify_email(
    subject,
    body
):

    email_text = (
        str(subject)
        + " "
        + str(body)
    )

    vector = vectorizer.transform(
        [email_text]
    )

    prediction = model.predict(
        vector
    )[0]

    probabilities = model.predict_proba(
        vector
    )[0]

    classes = list(
        model.classes_
    )

    spam_probability = 0.0
    ham_probability = 0.0

    for class_value, probability in zip(
        classes,
        probabilities
    ):

        class_name = str(
            class_value
        ).lower()

        if class_name in [
            "spam",
            "1",
            "true"
        ]:

            spam_probability = probability

        elif class_name in [
            "ham",
            "0",
            "false"
        ]:

            ham_probability = probability

    if (
        spam_probability == 0
        and
        ham_probability == 0
        and
        len(probabilities) == 2
    ):

        ham_probability = probabilities[0]
        spam_probability = probabilities[1]

    prediction_text = str(
        prediction
    ).lower()

    is_spam = prediction_text in [
        "spam",
        "1",
        "true"
    ]

    confidence = (
        spam_probability
        if is_spam
        else ham_probability
    )

    return {

        "prediction": prediction,

        "is_spam": is_spam,

        "spam_prob": spam_probability,

        "ham_prob": ham_probability,

        "confidence": confidence,

        "keywords": extract_keywords(
            email_text
        )
    }


# ============================================================
# CLEAR MANUAL INPUT
# ============================================================

def clear_manual():

    st.session_state.manual_sender = ""

    st.session_state.manual_subject = ""

    st.session_state.manual_body = ""

    st.session_state.manual_result = None


# ============================================================
# DISPLAY GMAIL EMAIL
# ============================================================

def display_email(
    email,
    result,
    number
):

    subject = email.get(
        "subject",
        "No Subject"
    )

    sender = email.get(
        "sender",
        "Unknown Sender"
    )

    body = email.get(
        "body",
        ""
    )

    confidence = (
        result["confidence"] * 100
    )

    st.markdown(
        f"### 📧 Email {number}"
    )

    col1, col2 = st.columns(
        [2.4, 1]
    )

    with col1:

        st.markdown(
            f"*Subject:* {subject}"
        )

        st.markdown(
            f"*From:* {sender}"
        )

        preview = body.strip()

        if not preview:

            preview = (
                "No email content available."
            )

        st.markdown(
            f"*Preview:* {preview[:400]}"
        )

    with col2:

        if result["is_spam"]:

            st.error(
                "🚨 SPAM"
            )

        else:

            st.success(
                "✅ HAM"
            )

        st.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )

        st.progress(
            min(
                max(
                    result["confidence"],
                    0
                ),
                1
            )
        )

    with st.expander(
        "🔎 Details"
    ):

        detail_col1, detail_col2 = st.columns(
            2
        )

        with detail_col1:

            st.write(
                f"*Spam Probability:* "
                f"{result['spam_prob'] * 100:.2f}%"
            )

            st.write(
                f"*Ham Probability:* "
                f"{result['ham_prob'] * 100:.2f}%"
            )

        with detail_col2:

            if result["keywords"]:

                st.write(
                    "*Detected Keywords:*"
                )

                st.write(
                    ", ".join(
                        result["keywords"]
                    )
                )

            else:

                st.write(
                    "*Detected Keywords:* None"
                )

    st.divider()


# ============================================================
# MODEL GRAPHS
# ============================================================

def display_model_graphs():

    accuracy_exists = os.path.exists(
        ACCURACY_GRAPH
    )

    loss_exists = os.path.exists(
        LOSS_GRAPH
    )

    if not accuracy_exists and not loss_exists:
        return

    st.markdown(
        "#### 📈 Model Training Performance"
    )

    graph_col1, graph_col2 = st.columns(
        2
    )

    with graph_col1:

        if accuracy_exists:

            st.image(
                ACCURACY_GRAPH,
                caption="Training vs Validation Accuracy",
                width=430
            )

        else:

            st.info(
                "Accuracy graph not available."
            )

    with graph_col2:

        if loss_exists:

            st.image(
                LOSS_GRAPH,
                caption="Training vs Validation Loss",
                width=430
            )

        else:

            st.info(
                "Loss graph not available."
            )


# ============================================================
# HEADER
# ============================================================

header_col1, header_col2 = st.columns(
    [5, 1]
)

with header_col1:

    st.title(
        "🛡️ Email Spam Detection"
    )

    st.caption(
        "Intelligent spam detection powered by machine learning"
    )

with header_col2:

    dark_mode = st.toggle(
        "🌙 Dark",
        value=st.session_state.dark_mode
    )

    if dark_mode != st.session_state.dark_mode:

        st.session_state.dark_mode = dark_mode

        st.rerun()


# ============================================================
# SYSTEM STATUS
# ============================================================

st.success(
    "● System Online"
)


# ============================================================
# APPLICATION TABS
# ============================================================

manual_tab, gmail_tab = st.tabs(
    [
        "🔍 Manual Check",
        "📬 Gmail Inbox"
    ]
)


# ============================================================
# MANUAL CHECK
# ============================================================

with manual_tab:

    st.header(
        "🔍 Analyze an Email"
    )

    st.caption(
        "Enter an email subject and message to determine "
        "whether it is spam or legitimate."
    )

    # ========================================================
    # NEW: FROM / SENDER EMAIL
    # ========================================================

    sender = st.text_input(
        "From / Sender Email",
        placeholder="Example: sender@example.com",
        key="manual_sender"
    )

    # ========================================================
    # EXISTING: EMAIL SUBJECT
    # ========================================================

    subject = st.text_input(
        "Email Subject",
        placeholder=(
            "Example: Congratulations! You have won a prize"
        ),
        key="manual_subject"
    )

    # ========================================================
    # EXISTING: EMAIL CONTENT
    # ========================================================

    body = st.text_area(
        "Email Content",
        placeholder=(
            "Paste the complete email message here..."
        ),
        height=220,
        key="manual_body"
    )

    st.write("")

    button_col1, button_col2, empty_col = st.columns(
        [1.4, 1.0, 5]
    )

    with button_col1:

        analyze_button = st.button(
            "🔍 Analyze Email",
            type="primary",
            use_container_width=True
        )

    with button_col2:

        st.button(
            "🗑️ Clear",
            use_container_width=True,
            on_click=clear_manual
        )

    if analyze_button:

        if (
            not sender.strip()
            and
            not subject.strip()
            and
            not body.strip()
        ):

            st.warning(
                "⚠️ Please enter sender email, email subject, "
                "or email content."
            )

        else:

            with st.spinner(
                "Analyzing email..."
            ):

                # Existing model logic is unchanged.
                # Sender is collected for the interface,
                # but classification still uses subject + body.

                st.session_state.manual_result = (
                    classify_email(
                        subject,
                        body
                    )
                )

    if st.session_state.manual_result:

        result = st.session_state.manual_result

        st.divider()

        st.subheader(
            "📊 Classification Result"
        )

        result_col1, result_col2 = st.columns(
            [2.4, 1]
        )

        with result_col1:

            st.markdown(
                f"*From / Sender:* "
                f"{sender or 'Not provided'}"
            )

            st.markdown(
                f"*Subject:* "
                f"{subject or 'No Subject'}"
            )

            st.markdown(
                "*Message Preview:*"
            )

            if body.strip():

                st.write(
                    body[:600]
                )

            else:

                st.write(
                    "No message content provided."
                )

        with result_col2:

            if result["is_spam"]:

                st.error(
                    "🚨 SPAM"
                )

            else:

                st.success(
                    "✅ HAM"
                )

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.1f}%"
            )

            st.progress(
                min(
                    max(
                        result["confidence"],
                        0
                    ),
                    1
                )
            )

        with st.expander(
            "🔎 View Details"
        ):

            st.write(
                f"*Spam Probability:* "
                f"{result['spam_prob'] * 100:.2f}%"
            )

            st.write(
                f"*Ham Probability:* "
                f"{result['ham_prob'] * 100:.2f}%"
            )

            if result["keywords"]:

                st.write(
                    "*Detected Spam Keywords:*"
                )

                st.write(
                    ", ".join(
                        result["keywords"]
                    )
                )

            else:

                st.write(
                    "*Detected Spam Keywords:* None"
                )


# ============================================================
# GMAIL INBOX
# ============================================================

with gmail_tab:

    st.header(
        "📬 Gmail Security Center"
    )

    st.caption(
        "Connect your Gmail account and automatically "
        "scan inbox messages for spam."
    )

    col1, col2, col3 = st.columns(
        [1.4, 1.2, 1.4]
    )

    with col1:

        if st.button(
            "🔗 Connect to Gmail",
            use_container_width=True
        ):

            with st.spinner(
                "Connecting to Gmail..."
            ):

                st.session_state.gmail_creds = (
                    authenticate_gmail()
                )

            if st.session_state.gmail_creds:

                st.success(
                    "✅ Gmail connected successfully."
                )

    with col2:

        if st.button(
            "🔌 Disconnect",
            use_container_width=True
        ):

            if os.path.exists(
                TOKEN_FILE
            ):

                try:

                    os.remove(
                        TOKEN_FILE
                    )

                except Exception:

                    pass

            st.session_state.gmail_creds = None

            st.session_state.classified_emails = None

            st.success(
                "Gmail disconnected."
            )

    with col3:

        scan_button = st.button(
            "📥 Scan Inbox",
            type="primary",
            use_container_width=True
        )

    # ========================================================
    # CONNECTION STATUS
    # ========================================================

    if st.session_state.gmail_creds:

        st.info(
            "🟢 Gmail is connected."
        )

        # ====================================================
        # SCAN INBOX
        # ====================================================

        if scan_button:

            if st.session_state.gmail_creds is None:

                st.session_state.gmail_creds = authenticate_gmail()

            if st.session_state.gmail_creds:

                with st.spinner(
                    "Scanning Gmail inbox..."
                ):

                    emails = fetch_emails(
                        st.session_state.gmail_creds
                    )

                if emails:

                    classified_emails = []

                    for email in emails:

                        result = classify_email(
                            email["subject"],
                            email["body"]
                        )

                        classified_emails.append(
                            {
                                "email": email,
                                "result": result
                            }
                        )

                    st.session_state.classified_emails = (
                        classified_emails
                    )

    # ========================================================
    # DISPLAY SAVED SCAN RESULTS
    # ========================================================

    if st.session_state.classified_emails:

        classified_emails = (
            st.session_state.classified_emails
        )

        total_emails = len(
            classified_emails
        )

        spam_count = sum(
            1
            for item in classified_emails
            if item["result"]["is_spam"]
        )

        ham_count = (
            total_emails
            -
            spam_count
        )

        st.success(
            f"✅ Found {total_emails} emails"
        )

        # ====================================================
        # INBOX OVERVIEW
        # ====================================================

        st.header(
            "📊 Inbox Overview"
        )

        metric1, metric2, metric3 = st.columns(
            3
        )

        with metric1:

            st.metric(
                "📨 Emails Scanned",
                total_emails
            )

        with metric2:

            st.metric(
                "🚨 SPAM Detected",
                spam_count
            )

        with metric3:

            st.metric(
                "✅ HAM Detected",
                ham_count
            )

        # ====================================================
        # MODEL GRAPHS
        # ====================================================

        display_model_graphs()

        # ====================================================
        # FILTER
        # ====================================================

        st.header(
            "🔎 Filter Results"
        )

        filter_option = st.selectbox(
            "Filter Emails",
            [
                "All Emails",
                "Spam Only",
                "Ham Only"
            ]
        )

        filtered_emails = []

        for item in classified_emails:

            is_spam = item[
                "result"
            ]["is_spam"]

            if filter_option == "All Emails":

                filtered_emails.append(
                    item
                )

            elif (
                filter_option == "Spam Only"
                and
                is_spam
            ):

                filtered_emails.append(
                    item
                )

            elif (
                filter_option == "Ham Only"
                and
                not is_spam
            ):

                filtered_emails.append(
                    item
                )

        st.caption(
            f"Showing {len(filtered_emails)} "
            f"of {total_emails} emails"
        )

        # ====================================================
        # EMAIL ANALYSIS
        # ====================================================

        st.header(
            "📧 Email Analysis"
        )

        if filtered_emails:

            for index, item in enumerate(
                filtered_emails,
                start=1
            ):

                display_email(
                    item["email"],
                    item["result"],
                    index
                )

        else:

            st.info(
                "No emails match the selected filter."
            )

    else:

        st.info(
            "👆 Click 'Connect to Gmail' to start scanning your inbox."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ Email Spam Detection • "
    "TF-IDF + Machine Learning Spam Detection"
)
