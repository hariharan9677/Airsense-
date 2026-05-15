import os
import gc          # garbage collector — frees unused memory on 8 GB RAM
import requests

# ── Must be set BEFORE importing matplotlib or numpy ────────────────────────
# i5-12450H has 8 cores; cap threads to 4 so RAM isn't over-committed
os.environ.setdefault("OMP_NUM_THREADS",     "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS",     "4")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves ~30 MB vs TkAgg
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ─────────────────────────────────────────────
# BACKGROUND IMAGE HELPER
# ─────────────────────────────────────────────
def get_base64_image(image_path):
    """Convert a local image file to base64 string for CSS embedding."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

def set_background(image_path=None, image_url=None, overlay_opacity=0.55):
    """
    Set a background image for the Streamlit app.
    
    Args:
        image_path (str): Path to a local image file (e.g., "background.jpg")
        image_url  (str): URL of an online image (used if image_path is None or not found)
        overlay_opacity (float): Dark overlay opacity 0.0–1.0 (default 0.55)
    
    Usage examples:
        set_background(image_path="background.jpg")
        set_background(image_url="https://images.unsplash.com/photo-1519692933481-e162a57d6721?w=1920")
        set_background(image_path="bg.jpg", overlay_opacity=0.4)
    """
    bg_css = ""

    # Try local file first
    if image_path:
        b64 = get_base64_image(image_path)
        if b64:
            ext = image_path.rsplit(".", 1)[-1].lower()
            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
            bg_css = f'background-image: url("data:image/{mime};base64,{b64}");'

    # Fall back to URL if no local image
    if not bg_css and image_url:
        bg_css = f'background-image: url("{image_url}");'

    # Default fallback: dark gradient (no image)
    if not bg_css:
        bg_css = "background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);"

    st.markdown(f"""
    <style>
        /* ── Main app background ── */
        .stApp {{
            {bg_css}
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}

        /* ── Dark overlay so text stays readable ── */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0, 5, 20, {overlay_opacity});
            z-index: 0;
            pointer-events: none;
        }}

        /* ── Sidebar gets a matching semi-transparent look ── */
        [data-testid="stSidebar"] {{
            background: rgba(5, 15, 35, 0.82) !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}

        /* ── All main content sits above the overlay ── */
        .main .block-container {{
            position: relative;
            z-index: 1;
        }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# EMAIL ALERT FUNCTION
# ─────────────────────────────────────────────
def send_email_alert(recipient_email, sender_email, sender_password, subject, body):
    """Send an email alert using Gmail SMTP."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender_email
        msg["To"]      = recipient_email

        html_body = f"""
        <html><body>
        <div style="font-family:Arial,sans-serif; padding:20px; background:#f4f4f4; border-radius:10px;">
            <h2 style="color:#cc0000;">⚠️ AQIS Alert Notification</h2>
            <div style="background:white; padding:15px; border-radius:8px; border-left:5px solid #cc0000;">
                {body.replace(chr(10), '<br>')}
            </div>
            <p style="color:#888; font-size:12px; margin-top:15px;">
                This is an automated alert from the Air Quality Intelligence System (AQIS).
            </p>
        </div>
        </body></html>
        """
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        return True, "✅ Alert email sent successfully!"
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Authentication failed. Use a Gmail App Password (not your regular password)."
    except Exception as e:
        return False, f"❌ Failed to send email: {e}"


def check_alerts(input_values, prediction, place):
    """Check all alert conditions and return list of triggered alerts."""
    alerts = []
    temp     = input_values.get("Temperature", 0)
    humidity = input_values.get("Humidity", 0)
    pm25     = input_values.get("PM2.5", 0)
    co       = input_values.get("CO", 0)

    if temp > 40:
        alerts.append(("🌡️ Extreme Heat",      f"Temperature is critically HIGH at {temp:.1f}°C in {place}. Heat stroke risk!"))
    elif temp < 5:
        alerts.append(("🥶 Extreme Cold",       f"Temperature is critically LOW at {temp:.1f}°C in {place}. Hypothermia risk!"))
    if prediction in ["Poor", "Hazardous"]:
        alerts.append(("🌫️ Poor Air Quality",   f"Air Quality predicted as '{prediction}' in {place}. Avoid outdoor exposure!"))
    if pm25 > 150:
        alerts.append(("💨 High PM2.5",         f"PM2.5 level is dangerously HIGH at {pm25:.1f} µg/m³ in {place}."))
    if co > 15:
        alerts.append(("☠️ High CO Level",      f"Carbon Monoxide is HIGH at {co:.1f} ppm in {place}. Ventilate immediately!"))
    if humidity > 90:
        alerts.append(("💧 Very High Humidity", f"Humidity is very HIGH at {humidity:.1f}% in {place}. Mold and heat stress risk."))

    return alerts


# ─────────────────────────────────────────────
# PERMANENT STORAGE HELPERS
# ─────────────────────────────────────────────
HISTORY_FILE = "prediction_history.csv"
HISTORY_COLS = ["Date", "Time", "Place", "Result"]

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=HISTORY_COLS)

def save_history(df_history):
    try:
        df_history.to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        st.warning(f"⚠️ Could not save history: {e}")

# ─────────────────────────────────────────────
# MUNICIPAL TEAM STORAGE HELPERS
# ─────────────────────────────────────────────
TEAM_FILE = "municipal_team.csv"
TEAM_COLS = ["Name", "Role", "Email", "Phone"]

def load_team():
    if os.path.exists(TEAM_FILE):
        try:
            return pd.read_csv(TEAM_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=TEAM_COLS)

def save_team(df_team):
    try:
        df_team.to_csv(TEAM_FILE, index=False)
    except Exception as e:
        st.warning(f"⚠️ Could not save team: {e}")

def send_alert_to_team(team_df, sender_email, sender_password, subject, body):
    results = []
    active_members = team_df[team_df["Email"].notna() & (team_df["Email"] != "")]
    for _, member in active_members.iterrows():
        success, msg = send_email_alert(
            member["Email"], sender_email, sender_password, subject,
            f"Dear {member['Name']} ({member['Role']}),\n\n{body}"
        )
        results.append((member["Name"], member["Role"], member["Email"], success, msg))
    return results


# ─────────────────────────────────────────────
# 1. PAGE CONFIG & STYLING
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 1. PAGE CONFIG & STYLING
# ─────────────────────────────────────────────
st.set_page_config(page_title="AQIS - Air Quality Intelligence System", layout="wide")

# ✅ APPLY BACKGROUND IMAGE
set_background(
    image_path=r"C:\Users\acer\Desktop\project weather ass\air12.png",
    overlay_opacity=0.30
)
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    :root { --card-bg: rgba(255,255,255,0.06); --card-border: rgba(255,255,255,0.08); }
    .stApp { padding: 18px 24px; }
    .login-box { 
        max-width: 400px; margin: auto; padding: 2rem; border-radius: 15px; 
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .hero-section {
        background-color: rgba(51, 102, 204, 0.15); padding: 30px; border-radius: 15px; 
        border-left: 6px solid #3366CC; margin-bottom: 25px;
        backdrop-filter: blur(8px);
    }
    .home-hero {
        text-align: center; padding: 50px; background: rgba(255,255,255,0.02);
        border-radius: 20px; border: 1px solid rgba(255,255,255,0.1);
    }
    .aq-badge { 
        padding: 10px 18px; border-radius: 12px; font-weight: 700; 
        display: inline-block; border: 2px solid transparent; margin-bottom: 15px;
    }
    .aq-good     { background: #d1fae5; border-color: #16a34a; color: #064e3b; }
    .aq-moderate { background: #facc1533; border-color: #ecff1a; color: #ecff1a; }
    .aq-poor     { background: #fed7aa; border-color: #ea580c; color: #7c2d12; }
    .aq-hazardous{ background: #ef444433; border-color: #991b1b; color: #6b0b0b; }
    .info-card { 
        background: rgba(255,255,255,0.06); border: 1px solid var(--card-border); 
        border-radius: 14px; padding: 14px; min-width: 260px; margin-bottom: 12px;
        backdrop-filter: blur(6px);
    }
    .info-title { font-size: 15px; margin: 0 0 6px 0; font-weight: 600; color: inherit; }
    .info-value { font-size: 17px; margin: 0; font-weight: 700; color: inherit; }
    .scroll-row { display: flex; gap: 16px; overflow-x: auto; padding-bottom: 8px; }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: rgba(10, 20, 40, 0.6) !important;
        border-radius: 10px;
    }
    div[data-testid="metric-container"] {
        background: rgba(51, 102, 204, 0.15);
        border: 1px solid rgba(51, 102, 204, 0.3);
        border-radius: 10px;
        padding: 10px 16px;
        backdrop-filter: blur(6px);
    }
</style>
""", unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    :root { --card-bg: rgba(255,255,255,0.06); --card-border: rgba(255,255,255,0.08); }
    .stApp { padding: 18px 24px; }
    .login-box { 
        max-width: 400px; margin: auto; padding: 2rem; border-radius: 15px; 
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .hero-section {
        background-color: rgba(51, 102, 204, 0.15); padding: 30px; border-radius: 15px; 
        border-left: 6px solid #3366CC; margin-bottom: 25px;
        backdrop-filter: blur(8px);
    }
    .home-hero {
        text-align: center; padding: 50px; background: rgba(255,255,255,0.02);
        border-radius: 20px; border: 1px solid rgba(255,255,255,0.1);
    }
    .aq-badge { 
        padding: 10px 18px; border-radius: 12px; font-weight: 700; 
        display: inline-block; border: 2px solid transparent; margin-bottom: 15px;
    }
    .aq-good     { background: #d1fae5; border-color: #16a34a; color: #064e3b; }
    .aq-moderate { background: #facc1533; border-color: #ecff1a; color: #ecff1a; }
    .aq-poor     { background: #fed7aa; border-color: #ea580c; color: #7c2d12; }
    .aq-hazardous{ background: #ef444433; border-color: #991b1b; color: #6b0b0b; }
    .info-card { 
        background: rgba(255,255,255,0.06); border: 1px solid var(--card-border); 
        border-radius: 14px; padding: 14px; min-width: 260px; margin-bottom: 12px;
        backdrop-filter: blur(6px);
    }
    .info-title { font-size: 15px; margin: 0 0 6px 0; font-weight: 600; color: inherit; }
    .info-value { font-size: 17px; margin: 0; font-weight: 700; color: inherit; }
    .scroll-row { display: flex; gap: 16px; overflow-x: auto; padding-bottom: 8px; }

    /* Make Streamlit native elements blend with the background */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: rgba(10, 20, 40, 0.6) !important;
        border-radius: 10px;
    }
    div[data-testid="metric-container"] {
        background: rgba(51, 102, 204, 0.15);
        border: 1px solid rgba(51, 102, 204, 0.3);
        border-radius: 10px;
        padding: 10px 16px;
        backdrop-filter: blur(6px);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. DATA & MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    try:
        df = pd.read_csv(r"air.csv")
        df.columns = df.columns.str.strip()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        df = df.dropna(subset=['Air Quality'])
        df['Air Quality'] = df['Air Quality'].astype(str).str.strip().str.title()

        features = [
            "Temperature", "Humidity", "PM2.5", "PM10",
            "NO2", "SO2", "CO",
            "Proximity_to_Industrial_Areas", "Population_Density"
        ]
        X = df[features]
        y = df['Air Quality']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.18, random_state=42,
            stratify=y if len(y.unique()) > 1 else None
        )
        # ── Optimized for 8GB RAM / i5-12450H (8-core) ──────────────
        # n_estimators : 80  → was 150; saves ~45% RAM, still accurate
        # n_jobs       : 4   → uses 4 of 8 cores; leaves RAM for OS/browser
        # max_depth    : 20  → caps tree depth to cut memory per tree
        # max_features : "sqrt" → default but explicit; limits split search
        # min_samples_split: 4 → slightly larger splits = smaller trees
        clf = RandomForestClassifier(
            n_estimators=80,
            n_jobs=4,
            max_depth=20,
            max_features="sqrt",
            min_samples_split=4,
            random_state=42
        )
        clf.fit(X_train, y_train)

        # Free training data from RAM immediately — important on 8 GB systems
        del X_train, y_train
        gc.collect()

        return clf, df, features, X_test, y_test

    except FileNotFoundError:
        st.error("❌ 'air.csv' not found. Place it in the same folder as this script.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None


# ─────────────────────────────────────────────
# 3. SAFETY HELPER FUNCTIONS
# ─────────────────────────────────────────────
def mask_recommendation(aqi):
    return {
        "Good":      "No mask needed",
        "Moderate":  "Surgical mask recommended",
        "Poor":      "N95 mask required",
        "Hazardous": "Avoid going outside",
    }.get(aqi, "Unknown")


def pregnant_safety(aqi, so2=0, no2=0):
    if aqi == "Good" and so2 < 20 and no2 < 40:
        return "Safe for short outdoor exposure"
    elif aqi == "Moderate":
        return "Limit outdoor time"
    elif aqi == "Poor":
        return "Unsafe – avoid outdoors"
    elif aqi == "Hazardous":
        return "Highly dangerous – stay indoors"
    return "Unknown"


def kids_safety(aqi):
    return {
        "Good":      "Safe for school / outdoor play",
        "Moderate":  "Caution for prolonged outdoor play",
        "Poor":      "Avoid heavy outdoor activity",
        "Hazardous": "Not safe for kids – stay indoors",
    }.get(aqi, "Unknown")


def senior_safety(aqi):
    return {
        "Good":      "Safe",
        "Moderate":  "Monitor health",
        "Poor":      "Caution – risk of irritation",
        "Hazardous": "Dangerous – stay indoors",
    }.get(aqi, "Unknown")


def asthma_risk(aqi):
    return {
        "Good":      "Low risk",
        "Moderate":  "Moderate risk – keep inhaler handy",
        "Poor":      "High risk – limit all exertion",
        "Hazardous": "Emergency risk – stay indoors",
    }.get(aqi, "Unknown")


def exercise_advice(aqi):
    return {
        "Good":      "Perfect conditions for exercise",
        "Moderate":  "Avoid heavy cardio outdoors",
        "Poor":      "Exercise indoors only",
        "Hazardous": "No outdoor activity",
    }.get(aqi, "Unknown")
def geocode_place(place_name):
    """Convert a place name to (lat, lon) using OpenStreetMap Nominatim."""
    try:
        url    = "https://nominatim.openstreetmap.org/search"
        params = {"q": place_name, "format": "json", "limit": 1}
        headers = {"User-Agent": "AQIS-App/1.0"}
        resp   = requests.get(url, params=params, headers=headers, timeout=5)
        data   = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None

# ─────────────────────────────────────────────
# MUNICIPAL TEAM PAGE
# ─────────────────────────────────────────────
def show_municipal_team():
    st.title("🏛️ Municipal Team Management")
    st.caption("Add team members here. They will automatically receive email alerts when dangerous conditions are detected.")

    team_df = load_team()
    if "team" not in st.session_state:
        st.session_state["team"] = team_df

    st.subheader("➕ Add New Team Member")
    with st.form("add_member_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name  = st.text_input("Full Name *", placeholder="e.g. Dr. Ramesh Kumar")
            role  = st.selectbox("Role / Department", [
                "Municipal Commissioner", "Health Officer", "Environment Officer",
                "Pollution Control Officer", "Fire & Emergency Services",
                "Public Works Department", "Mayor's Office", "Other"
            ])
        with col2:
            email = st.text_input("Email Address *", placeholder="officer@municipality.gov.in")
            phone = st.text_input("Phone Number", placeholder="+91 XXXXXXXXXX")

        submitted = st.form_submit_button("➕ Add Member", use_container_width=True)
        if submitted:
            if not name or not email:
                st.error("❌ Name and Email are required.")
            elif email in st.session_state["team"]["Email"].values:
                st.warning("⚠️ This email is already in the team list.")
            else:
                new_member = {"Name": name, "Role": role, "Email": email, "Phone": phone}
                st.session_state["team"] = pd.concat(
                    [st.session_state["team"], pd.DataFrame([new_member])], ignore_index=True
                )
                save_team(st.session_state["team"])
                st.success(f"✅ {name} ({role}) added to the municipal team!")
                st.rerun()

    st.write("---")
    st.subheader("👥 Current Municipal Team")
    if st.session_state["team"].empty:
        st.info("No team members added yet. Add members above to enable automatic alerts.")
    else:
        st.dataframe(st.session_state["team"], use_container_width=True)
        st.caption(f"Total members: **{len(st.session_state['team'])}** — All will receive alerts automatically when prediction runs.")

        st.subheader("🗑️ Remove a Member")
        member_to_remove = st.selectbox("Select member to remove:", options=st.session_state["team"]["Name"].tolist())
        if st.button("Remove Selected Member", type="secondary"):
            st.session_state["team"] = st.session_state["team"][
                st.session_state["team"]["Name"] != member_to_remove
            ].reset_index(drop=True)
            save_team(st.session_state["team"])
            st.success(f"✅ {member_to_remove} removed from team.")
            st.rerun()

        st.write("---")
        csv_team = st.session_state["team"].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Team List as CSV", csv_team, "municipal_team.csv", "text/csv")

    st.write("---")
    st.subheader("🧪 Send Test Alert to All Members")
    st.caption("Use this to verify all team members receive emails correctly.")
    with st.expander("⚙️ Sender Gmail Settings for Test"):
        t_sender   = st.text_input("Your Gmail (Sender)", key="test_sender", placeholder="yourname@gmail.com")
        t_password = st.text_input("Gmail App Password", key="test_pass", type="password")

    if st.button("📤 Send Test Alert to All Members", use_container_width=True):
        if st.session_state["team"].empty:
            st.warning("No team members to send to.")
        elif not t_sender or not t_password:
            st.warning("Please enter sender Gmail and App Password above.")
        else:
            test_body = (
                "This is a TEST alert from the Air Quality Intelligence System (AQIS).\n\n"
                "If you received this, your email is correctly registered and you will "
                "receive automatic alerts when dangerous air quality or temperature conditions are detected.\n\n"
                "No action is required for this test message."
            )
            results = send_alert_to_team(
                st.session_state["team"], t_sender, t_password,
                "🧪 AQIS Test Alert – Email Verification", test_body
            )
            for name, role, email, success, msg in results:
                if success:
                    st.success(f"✅ {name} ({role}) — {email}")
                else:
                    st.error(f"❌ {name} ({role}) — {email}: {msg}")


# ─────────────────────────────────────────────
# 4. PAGE FUNCTIONS
# ─────────────────────────────────────────────

def show_home():
    st.markdown(
        "<div class='hero-section'>"
        "<h1>🌍 AirSense Monitoring System</h1>"
        "<p>Protecting communities through real-time pollutant monitoring and Machine Learning insights.</p>"
        "</div>",
        unsafe_allow_html=True
    )

    st.write("### 📜 Project Mission")
    st.write(
        "This platform bridges the gap between complex environmental data and public health safety. "
        "By using 150+ Decision Trees in our Random Forest model, we provide high-accuracy predictions "
        "to keep you and your community safe."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 Major Pollutants Tracked")
        st.markdown("""
        - **PM2.5 & PM10:** Fine particulate matter that enters the lungs and bloodstream.
        - **Nitrogen Dioxide (NO2):** Primarily from vehicle emissions; causes respiratory issues.
        - **Sulfur Dioxide (SO2):** Produced by industrial burning; contributes to acid rain.
        - **Carbon Monoxide (CO):** Odorless gas that reduces oxygen delivery to organs.
        """)

    with col2:
        st.subheader("💡 Why Use This Tool?")
        st.markdown("""
        - **Plan Your Day:** Know when it's safe to exercise outdoors.
        - **Protect Vulnerable Groups:** Specific advice for kids, seniors & pregnant women.
        - **Track History:** Save predictions per location for trend analysis.
        - **Real-Time Analysis:** Instant ML results based on current inputs.
        """)

    st.write("---")
    st.subheader("🏥 Health Risk Reference Table")
    health_data = {
        "AQI Category": ["Good", "Moderate", "Poor", "Hazardous"],
        "Health Impact": [
            "Minimal impact – clean air",
            "Minor discomfort for sensitive groups",
            "Breathing discomfort on prolonged exposure",
            "Respiratory effects even on healthy people"
        ],
        "Recommended Action": [
            "No action needed – enjoy outdoors",
            "Sensitive people: avoid heavy outdoor work",
            "Wear N95 mask; limit outdoor time",
            "Stay indoors; use air purifiers"
        ]
    }
    st.table(pd.DataFrame(health_data))

    st.info("💡 **Pro Tip:** Use the **Prediction** tab in the sidebar to enter specific pollutant levels and get an instant AI classification.")

    st.write("---")
    st.write("### Ready to start?")
    if st.button("🚀 Launch Prediction", use_container_width=True):
        st.session_state["page"] = "Prediction"
        st.rerun()


def show_prediction(clf, features, df):
    st.title("🔮 AI Prediction & Safety Analysis")
    place = st.text_input("📍 Enter a real city name (e.g., Chennai, Delhi, Mumbai)", "Chennai")

    st.subheader("📧 Municipal Alert Settings")
    st.caption("Analysis report will be sent automatically to ALL registered municipal team members.")

    with st.expander("⚙️ Configure Sender Gmail", expanded=False):
        st.info("💡 Use a **Gmail App Password** — Google Account → Security → 2-Step Verification → App Passwords")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            sender_email    = st.text_input("Your Gmail Address (Sender)", placeholder="yourname@gmail.com")
            sender_password = st.text_input("Gmail App Password", type="password", placeholder="16-char app password")
        with col_e2:
            enable_alerts = st.checkbox("✅ Enable Auto-Alert to Municipal Team", value=False)
            team_count    = len(st.session_state.get("team", load_team()))
            if team_count > 0:
                st.success(f"👥 {team_count} team member(s) registered — will receive report automatically.")
            else:
                st.warning("⚠️ No team members registered. Go to **Municipal Team** page to add members.")

        alert_conditions = st.multiselect(
            "Trigger alert when:",
            ["🌡️ Temperature too High (>40°C)", "🥶 Temperature too Low (<5°C)",
             "🌫️ Poor / Hazardous Air Quality", "💨 High PM2.5 (>150)",
             "☠️ High CO Level (>15 ppm)", "💧 Very High Humidity (>90%)"],
            default=["🌡️ Temperature too High (>40°C)", "🥶 Temperature too Low (<5°C)",
                     "🌫️ Poor / Hazardous Air Quality"]
        )

    st.sidebar.divider()
    st.sidebar.subheader("🎚️ Environmental Parameters")
    input_values = {}
    for col in features:
        input_values[col] = st.sidebar.slider(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].median())
        )

    st.subheader("📋 Current Input Values")
    st.table(pd.DataFrame([input_values]))

    if st.button("Analyze Air Quality", use_container_width=True):
        input_df   = pd.DataFrame([input_values])
        prediction = clf.predict(input_df)[0]
        now        = datetime.now()

        new_row = {"Date": now.strftime("%Y-%m-%d"), "Time": now.strftime("%H:%M:%S"),
                   "Place": place, "Result": prediction}
        st.session_state["history"] = pd.concat(
            [st.session_state["history"], pd.DataFrame([new_row])], ignore_index=True
        )
        save_history(st.session_state["history"])

        # ── Auto Alert to Municipal Team ──────────────────────
        if enable_alerts and sender_email and sender_password:
            triggered = check_alerts(input_values, prediction, place)
            selected_keywords = [c.split("(")[0].strip() for c in alert_conditions]
            triggered = [a for a in triggered if any(kw in a[0] for kw in selected_keywords)]

            # Build full analysis email (always sent)
            alert_status = "⚠️ ALERT CONDITIONS TRIGGERED" if triggered else "✅ All Clear – No Alert Conditions"

            input_table = "\n".join([f"  • {k}: {v}" for k, v in input_values.items()])

            safety_summary = (
                f"  🎭 Mask:        {mask_recommendation(prediction)}\n"
                f"  🤰 Pregnant:    {pregnant_safety(prediction, input_values.get('SO2',0), input_values.get('NO2',0))}\n"
                f"  👶 Kids:        {kids_safety(prediction)}\n"
                f"  👴 Senior:      {senior_safety(prediction)}\n"
                f"  🫁 Asthma Risk: {asthma_risk(prediction)}\n"
                f"  🏃 Exercise:    {exercise_advice(prediction)}"
            )

            alert_details = ""
            if triggered:
                alert_details = "\n⚠️ Triggered Alerts:\n"
                for alert_title, alert_msg in triggered:
                    alert_details += f"  • {alert_title}: {alert_msg}\n"
            else:
                alert_details = "\n✅ No specific alert thresholds were crossed.\n"

            email_body = f"""
📍 Location  : {place}
📅 Date/Time : {now.strftime('%Y-%m-%d %H:%M:%S')}
🌍 Air Quality Prediction: {prediction}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 INPUT POLLUTANT VALUES:
{input_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{alert_status}
{alert_details}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛡️ SAFETY RECOMMENDATIONS:
{safety_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Please take appropriate action as per your department protocol.
"""

            subject = f"📊 AQIS Report: {prediction} Air Quality at {place} — {now.strftime('%Y-%m-%d %H:%M')}"

            current_team = st.session_state.get("team", load_team())
            if not current_team.empty:
                st.subheader("📤 Sending Analysis Report to Municipal Team...")
                results   = send_alert_to_team(current_team, sender_email, sender_password, subject, email_body)
                sent_ok   = [r for r in results if r[3]]
                sent_fail = [r for r in results if not r[3]]
                if sent_ok:
                    st.success(f"✅ Report sent to {len(sent_ok)} team member(s):")
                    for name, role, eml, _, _ in sent_ok:
                        st.markdown(f"  - **{name}** ({role}) → {eml}")
                if sent_fail:
                    st.error(f"❌ Failed for {len(sent_fail)} member(s):")
                    for name, role, eml, _, msg in sent_fail:
                        st.markdown(f"  - **{name}** ({role}) → {eml}: {msg}")
            else:
                st.warning("⚠️ No municipal team members registered. Go to **Municipal Team** page to add members.")

            # Show alert status on screen
            if triggered:
                st.warning("⚠️ **Alert Conditions Triggered:**")
                for alert_title, alert_msg in triggered:
                    st.markdown(f"- **{alert_title}:** {alert_msg}")
            else:
                st.info("✅ No alert conditions triggered. Full analysis report still sent to team.")

        elif enable_alerts and not (sender_email and sender_password):
            st.warning("⚠️ Please fill in Gmail and App Password in the Alert Settings above.")

        # ── Badge & Result ─────────────────────────────────────
        badge = {
            "Good": "aq-good", "Moderate": "aq-moderate",
            "Poor": "aq-poor",  "Hazardous": "aq-hazardous"
        }.get(prediction, "aq-unknown")
        st.markdown(
            f"## {place} Status: <span class='aq-badge {badge}'>{prediction}</span>",
            unsafe_allow_html=True
        )
        st.success(f"✅ Prediction recorded at {now.strftime('%H:%M:%S')} on {now.strftime('%Y-%m-%d')}")

        # ── 🗺️ WORLD MAP ──────────────────────────────────────
        st.subheader("🗺️ Location on World Map")

        aqi_colors = {
            "Good":      "#22c55e",
            "Moderate":  "#facc15",
            "Poor":      "#fb923c",
            "Hazardous": "#ef4444",
        }
        pin_color = aqi_colors.get(prediction, "#3366CC")

        with st.spinner("🌍 Locating place on map..."):
            lat, lon = geocode_place(place)

        if lat is not None:
            fig_map = go.Figure()
            fig_map.add_trace(go.Scattergeo(
                lat=[lat], lon=[lon], mode="markers",
                marker=dict(size=30, color=pin_color, opacity=0.2, symbol="circle"),
                showlegend=False, hoverinfo="skip"
            ))
            fig_map.add_trace(go.Scattergeo(
                lat=[lat], lon=[lon], mode="markers",
                marker=dict(size=20, color=pin_color, opacity=0.35, symbol="circle"),
                showlegend=False, hoverinfo="skip"
            ))
            fig_map.add_trace(go.Scattergeo(
                lat=[lat], lon=[lon],
                mode="markers+text",
                text=[f"  📍 {place}  |  {prediction}"],
                textfont=dict(color="white", size=13),
                textposition="middle right",
                marker=dict(size=14, color=pin_color, symbol="circle",
                            line=dict(width=2, color="white")),
                name=prediction,
                hovertemplate=(
                    f"<b>{place}</b><br>"
                    f"Air Quality: <b>{prediction}</b><br>"
                    f"Lat: {lat:.4f} | Lon: {lon:.4f}<extra></extra>"
                )
            ))
            fig_map.update_geos(
                projection_type="natural earth",
                showland=True,       landcolor="rgb(18, 30, 55)",
                showocean=True,      oceancolor="rgb(8, 18, 38)",
                showlakes=True,      lakecolor="rgb(8, 18, 38)",
                showrivers=True,     rivercolor="rgb(20, 40, 80)",
                showcountries=True,  countrycolor="rgba(255,255,255,0.15)",
                showcoastlines=True, coastlinecolor="rgba(255,255,255,0.3)",
                showframe=False,     bgcolor="rgba(0,0,0,0)",
                center=dict(lat=lat, lon=lon),
                projection_scale=4,
            )
            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                height=480, margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0.4)",
                            bordercolor="rgba(255,255,255,0.2)", borderwidth=1)
            )
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption(f"📌 Pinned: **{place}** — Coordinates: `{lat:.4f}°, {lon:.4f}°`")
        else:
            st.warning(f"⚠️ Could not locate **'{place}'** — try a specific city like **'Chennai'**, **'Delhi'**.")
            fig_map = go.Figure()
            fig_map.update_geos(
                projection_type="natural earth",
                showland=True,       landcolor="rgb(18, 30, 55)",
                showocean=True,      oceancolor="rgb(8, 18, 38)",
                showcountries=True,  countrycolor="rgba(255,255,255,0.15)",
                showcoastlines=True, coastlinecolor="rgba(255,255,255,0.3)",
                showframe=False,     bgcolor="rgba(0,0,0,0)",
            )
            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                height=400, margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)
        # ── end map ───────────────────────────────────────────

        # ── Safety Cards ──────────────────────────────────────
        st.subheader("🛡️ Personalized Safety Insights")
        safety_cards = [
            ("🎭 Mask Recommendation", mask_recommendation(prediction)),
            ("🤰 Pregnant Women",       pregnant_safety(prediction, input_values.get("SO2", 0), input_values.get("NO2", 0))),
            ("👶 Kids Safety",          kids_safety(prediction)),
            ("👴 Senior Citizens",      senior_safety(prediction)),
            ("🫁 Asthma Risk",          asthma_risk(prediction)),
            ("🏃 Exercise Advice",      exercise_advice(prediction)),
        ]
        st.markdown("<div class='scroll-row'>", unsafe_allow_html=True)
        for title, val in safety_cards:
            st.markdown(
                f"<div class='info-card' style='border:2px solid #3366CC'>"
                f"<div class='info-title'>{title}</div>"
                f"<div class='info-value'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────
        st.subheader("📊 Pollutant Impact on Prediction")
        fig = px.bar(
            x=features, y=clf.feature_importances_,
            labels={"x": "Feature", "y": "Importance Score"},
            title="Random Forest Feature Importance",
            color_discrete_sequence=["#3366CC"]
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🗂️ Dataset AQI Distribution")
        fig_pie = px.pie(
            df, names="Air Quality", hole=0.4,
            color="Air Quality",
            color_discrete_map={
                "Good": "#22c55e", "Moderate": "#facc15",
                "Poor": "#fb923c", "Hazardous": "#ef4444"
            }
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_pie, use_container_width=True)
def show_dashboard(clf, X_test, y_test):
    st.title("📊 Model Performance Dashboard")

    # ── Accuracy ──
    y_pred   = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Model Accuracy",   f"{accuracy:.2f}%")
    col2.metric("🌲 Trees in Forest",  clf.n_estimators)
    col3.metric("📋 Test Samples",     len(y_test))

    st.write("---")

    # ── Confusion Matrix ──
    st.subheader("🔢 Confusion Matrix")
    labels = sorted(y_test.unique())
    cm     = confusion_matrix(y_test, y_pred, labels=labels)
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix — Random Forest",
        color_continuous_scale="Blues"
    )
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0.3)",
        font_color   ="white"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.write("---")

    # ── Feature Importance ──
    st.subheader("📊 Feature Importance")
    fig_fi = px.bar(
        x=clf.feature_names_in_, y=clf.feature_importances_,
        labels={"x": "Feature", "y": "Importance Score"},
        title="Which pollutants matter most?",
        color_discrete_sequence=["#3366CC"]
    )
    fig_fi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0.3)",
        font_color   ="white"
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.write("---")

    # ── AQI Distribution ──
    st.subheader("🗂️ Dataset AQI Distribution")
    fig_pie = px.pie(
        names=y_test.value_counts().index,
        values=y_test.value_counts().values,
        hole=0.4,
        color=y_test.value_counts().index,
        color_discrete_map={
            "Good":      "#22c55e",
            "Moderate":  "#facc15",
            "Poor":      "#fb923c",
            "Hazardous": "#ef4444"
        }
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.write("---")

    # ── Prediction History ──
    st.subheader("📜 Prediction History")
    history_df = st.session_state.get("history", pd.DataFrame())
    if history_df.empty:
        st.info("No predictions recorded yet. Run a prediction first.")
    else:
        st.dataframe(history_df, use_container_width=True)
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download History as CSV",
            csv, "prediction_history.csv", "text/csv"
        )
# ─────────────────────────────────────────────
# 5. SESSION STATE INIT
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 5. SESSION STATE INIT
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = load_history()

if "team" not in st.session_state:
    st.session_state["team"] = load_team()

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "page" not in st.session_state:
    st.session_state["page"] = "Home"


# ─────────────────────────────────────────────
# 6. LOGIN SCREEN
# ─────────────────────────────────────────────
if not st.session_state["logged_in"]:
    _, col, _ = st.columns([1, 2, 1])
    with col:
        # st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.title(" Airsense🍃🌍 ")
        st.write("Enter your credentials to access the analytics platform.")
        with st.form("login_form"):
            user = st.text_input("Username")
            pw   = st.text_input("Password", type="password")
            if st.form_submit_button("Access System", use_container_width=True):
                if user == "admin" and pw == "admin123":
                    st.session_state["logged_in"] = True
                    st.session_state["page"] = "Home"
                    st.success("Login Successful!")
                    st.rerun()
                else:
                    st.error("Access Denied – Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 7. MAIN APP (LOGGED IN)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 7. MAIN APP (LOGGED IN)
# ─────────────────────────────────────────────
else:
    clf, df, features, X_test, y_test = load_model_and_data()

    if clf is not None:

        # ── Sidebar Logo / Title ──
        st.sidebar.markdown("""
        <h2 style='text-align:center; color:#4FC3F7;'>
            🌍 AirSense
        </h2>
        """, unsafe_allow_html=True)

        st.sidebar.write(" Welcome to AirSense Monitoring System")

        # ── Navigation Pages ──
        pages = [
            "🏠 Welcome",
            "🤖 AI Prediction",
            "📉 Dashboard",
            "👨‍💼 Authority Panel"
        ]

        # ── Session State Default ──
        if "page" not in st.session_state:
            st.session_state["page"] = pages[0]

        # ── Sidebar Navigation ──
        choice = st.sidebar.radio(
    "📌 Navigation",
    pages,
    index=pages.index(st.session_state["page"])
    if st.session_state["page"] in pages else 0
)
        st.session_state["page"] = choice

        st.sidebar.divider()

        # ── Logout Button ──
        if st.sidebar.button("🔓 Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.rerun()

        # ─────────────────────────────────────────────
        # PAGE NAVIGATION
        # ─────────────────────────────────────────────

        if choice == "🏠 Welcome":
            show_home()

        elif choice == "🤖 AI Prediction":
            show_prediction(clf, features, df)

        elif choice == "📉 Dashboard":
            show_dashboard(clf, X_test, y_test)

        elif choice == "👨‍💼 Authority Panel":
            show_municipal_team()
