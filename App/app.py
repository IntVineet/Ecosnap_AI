import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2
from pathlib import Path
from datetime import date
import requests

# ------------------------------------------
# BASIC CONFIG
# ------------------------------------------
st.set_page_config(
    page_title="EcoSnap • AI Waste Detector",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "waste_model_merged.keras"
LABEL_MAP_PATH = BASE_DIR / "label_map_merged.json"
ECO_DB_PATH = BASE_DIR / "eco_database.json"
UPCYCLE_PATH = BASE_DIR / "upcycle_ideas.json"
USER_DATA_PATH = BASE_DIR / "user_data.json"
IMG_SIZE = 224

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

# ------------------------------------------
# LOAD MODEL & DATA
# ------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_label_map():
    with open(LABEL_MAP_PATH, "r") as f:
        m = json.load(f)
    # convert keys to int if needed
    return {int(k): v for k, v in m.items()}

@st.cache_data
def load_eco_db():
    if ECO_DB_PATH.exists():
        with open(ECO_DB_PATH, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_upcycle_db():
    if UPCYCLE_PATH.exists():
        with open(UPCYCLE_PATH, "r") as f:
            return json.load(f)
    return {}

def load_user_data():
    default = {
        "total_points": 0,
        "total_scans": 0,
        "today_scans": 0,
        "last_scan_date": None,
        "streak_days": 0,
        "badges": [],
        "scan_history": []
    }
    if USER_DATA_PATH.exists():
        try:
            with open(USER_DATA_PATH, "r") as f:
                data = json.load(f)
        except Exception:
            return default
        # ensure all keys exist
        for k, v in default.items():
            if k not in data:
                data[k] = v
        return data
    return default

def save_user_data(data):
    with open(USER_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

# ------------------------------------------
# PREDICTION
# ------------------------------------------
def predict_image(img):
    model = load_model()
    label_map = load_label_map()

    # PIL → np.array
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img)
    elif isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    else:
        raise ValueError("Unsupported image type")

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100.0

    class_name = label_map.get(class_id, f"Class {class_id}")
    return class_name, confidence

# ------------------------------------------
# GAMIFICATION
# ------------------------------------------
def compute_points(confidence):
    return 10 + int(confidence // 10)

def update_user_stats(class_name, confidence):
    data = load_user_data()

    today = date.today().isoformat()
    last = data.get("last_scan_date")

    # streak
    if last is None:
        data["streak_days"] = 1
    else:
        last_date = date.fromisoformat(last)
        diff = (date.today() - last_date).days
        if diff == 1:
            data["streak_days"] += 1
        elif diff > 1:
            data["streak_days"] = 1

    data["last_scan_date"] = today

    # today scans
    if last != today:
        data["today_scans"] = 0
    data["today_scans"] += 1

    # points
    pts = compute_points(confidence)
    data["total_points"] += pts
    data["total_scans"] += 1

    # history
    data["scan_history"].append(
        {
            "date": today,
            "class_name": class_name,
            "confidence": round(confidence, 2),
            "points": pts,
        }
    )

    # badges
    if data["total_scans"] >= 1 and "First Scan" not in data["badges"]:
        data["badges"].append("First Scan")
    if data["total_scans"] >= 5 and "Eco Beginner" not in data["badges"]:
        data["badges"].append("Eco Beginner")
    if data["total_scans"] >= 15 and "Recycler" not in data["badges"]:
        data["badges"].append("Recycler")
    if data["total_scans"] >= 50 and "Green Hero" not in data["badges"]:
        data["badges"].append("Green Hero")

    save_user_data(data)
    return pts, data

# ------------------------------------------
# GROQ ECO CHAT
# ------------------------------------------
def groq_chat(msg, last_detection=None):
    if not GROQ_API_KEY:
        return "Groq API key missing. Add it in .streamlit/secrets.toml"

    system_prompt = (
        "You are EcoSnap AI — a helpful assistant for waste, recycling, "
        "and environmental impact. Be clear, short, and practical."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if last_detection:
        cls = last_detection.get("class_name")
        info = last_detection.get("info", {})
        ctx = (
            f"User last scanned: {cls}. "
            f"Decomposition: {info.get('decomposition', 'N/A')}. "
            f"Impact: {info.get('impact', 'N/A')}."
        )
        messages.append({"role": "system", "content": ctx})

    messages.append({"role": "user", "content": msg})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.6,
    }

    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions",
                            headers=headers, json=payload)
        data = res.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        if "error" in data:
            return "Groq Error: " + data["error"]["message"]
        return f"Unexpected response: {data}"
    except Exception as e:
        return f"Request failed: {e}"

# ------------------------------------------
# SESSION STATE INIT
# ------------------------------------------
if "nav" not in st.session_state:
    st.session_state["nav"] = "Home"

if "last_detection" not in st.session_state:
    st.session_state["last_detection"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ------------------------------------------
# UI HELPERS
# ------------------------------------------
def render_header():
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0;'>
            <h1>🌿 <span style='color:#16a34a'>EcoSnap</span> – AI Waste Detector</h1>
            <p style='color:#9ca3af;'>Snap ➝ Learn ➝ Reuse ➝ Save the Planet</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------
# PAGE: HOME
# ------------------------------------------
def page_home():
    render_header()

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.subheader("Why EcoSnap?")
        st.write(
            """
            EcoSnap turns your camera into an eco-coach:  
            • Identify what kind of waste you’re looking at  
            • Learn how long it takes to decompose  
            • Understand its environmental impact  
            • Get recycling tips and upcycle ideas  
            • Earn eco-points and badges as you scan  
            """
        )
        if st.button("🚀 Start Detection", use_container_width=True):
            st.session_state["nav"] = "Detection"

    with col2:
        st.subheader("Your Quick Stats")
        data = load_user_data()
        c1, c2, c3 = st.columns(3)
        c1.metric("Eco Points", data["total_points"])
        c2.metric("Total Scans", data["total_scans"])
        c3.metric("Streak (days)", data["streak_days"])

        st.markdown("---")
        st.subheader("About Developer")
        st.info("👨‍💻 **Vineet Kumar** · B.Tech CSE · Building AI for sustainability.")

# ------------------------------------------
# RESULT RENDERING
# ------------------------------------------
def show_results(pred, conf):
    eco_db = load_eco_db()
    up_db = load_upcycle_db()

    pts, data = update_user_stats(pred, conf)

    st.success(f"⭐ +{pts} Eco Points Earned!")
    st.write(f"### {pred} ({conf:.1f}% confidence)")

    if pred in eco_db:
        info = eco_db[pred]
        st.subheader("🕒 Decomposition Time")
        st.info(info.get("decomposition", "No decomposition data available."))

        st.subheader("⚠ Environmental Impact")
        st.warning(info.get("impact", "No impact data available."))

        st.subheader("♻ Recycling Tips")
        st.success(info.get("tips", "No recycling tips available."))
    else:
        st.warning("No environmental database entry for this item.")

    if pred in up_db:
        st.subheader("🎨 Upcycle Ideas")
        ideas = up_db[pred]
        # support list of strings or dicts
        if isinstance(ideas, list):
            for idea in ideas:
                if isinstance(idea, dict):
                    title = idea.get("title", "Idea")
                    desc = idea.get("description", "")
                    st.write(f"• **{title}** – {desc}")
                else:
                    st.write(f"• {idea}")
        else:
            st.write(str(ideas))

    st.session_state["last_detection"] = {
        "class_name": pred,
        "confidence": conf,
        "info": eco_db.get(pred, {}),
    }

# ------------------------------------------
# PAGE: DETECTION
# ------------------------------------------
def page_detection():
    render_header()
    st.subheader("🧪 AI Waste Detection")

    mode = st.radio(
        "Choose Mode",
        ["📸 Live Camera", "📁 Upload Photo"],
        horizontal=True
    )

    st.markdown("---")

    # Live camera mode
    if mode == "📸 Live Camera":
        img_file = st.camera_input("Capture a waste image")
        if img_file is not None:
            pil = Image.open(img_file)
            st.image(pil, caption="Captured Image", use_column_width=True)

            with st.spinner("Analyzing with AI..."):
                pred, conf = predict_image(pil)

            show_results(pred, conf)

    # Upload mode
    else:
        uploaded = st.file_uploader(
            "Upload an image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded is not None:
            pil = Image.open(uploaded)
            st.image(pil, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Analyzing with AI..."):
                pred, conf = predict_image(pil)

            show_results(pred, conf)

# ------------------------------------------
# PAGE: DASHBOARD
# ------------------------------------------
def page_dashboard():
    render_header()
    st.subheader("📊 Eco Dashboard")

    data = load_user_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eco Points", data["total_points"])
    c2.metric("Scans Today", data["today_scans"])
    c3.metric("Total Scans", data["total_scans"])
    level = data["total_points"] // 100 + 1
    c4.metric("Level", level)

    st.markdown("---")

    st.subheader("Achievements")
    if data["badges"]:
        for b in data["badges"]:
            st.success(f"🏅 {b}")
    else:
        st.info("No badges yet. Start scanning to earn some!")

    st.markdown("---")
    st.subheader("Recent Scans")
    hist = data.get("scan_history", [])[-10:]
    if not hist:
        st.write("No scans yet.")
    else:
        for item in reversed(hist):
            st.write(
                f"- {item['date']}: **{item['class_name']}** "
                f"({item['confidence']}%) – {item['points']} pts"
            )

# ------------------------------------------
# PAGE: ECO CHAT
# ------------------------------------------
def page_chat():
    render_header()
    st.subheader("💬 EcoSnap AI Assistant")

    last = st.session_state.get("last_detection")

    if last:
        st.info(
            f"Last scanned: **{last['class_name']}** "
            f"({last['confidence']:.1f}% confidence)"
        )
    else:
        st.info("No detection yet. Scan an item on the Detection tab for better context.")

    for role, text in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.write(text)

    user_msg = st.chat_input("Ask about recycling, waste, or eco-friendly habits...")
    if user_msg:
        st.session_state["chat_history"].append(("user", user_msg))
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = groq_chat(user_msg, last)
            st.write(reply)
        st.session_state["chat_history"].append(("assistant", reply))

# ------------------------------------------
# NAVIGATION
# ------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 EcoSnap")
    choice = st.radio(
        "Navigate",
        ["Home", "Detection", "Dashboard", "Eco Chat"],
        index=["Home", "Detection", "Dashboard", "Eco Chat"].index(
            st.session_state["nav"]
        ),
    )
    st.session_state["nav"] = choice

if st.session_state["nav"] == "Home":
    page_home()
elif st.session_state["nav"] == "Detection":
    page_detection()
elif st.session_state["nav"] == "Dashboard":
    page_dashboard()
elif st.session_state["nav"] == "Eco Chat":
    page_chat()
