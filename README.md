# 🌿 EcoSnap – AI Waste Detection & Eco Assistant
Snap → Learn → Reuse → Save the Planet
An AI-powered system to classify waste, learn environmental impact, get recycling tips, earn eco-points, and chat with an eco assistant — all in one app.

# 📌 Overview

EcoSnap is a full-stack AI project that turns your device into a real-time eco-coach.
Using a TensorFlow CNN model, camera detection, and Groq LLaMA-3.3, the app identifies waste items and instantly provides:

Decomposition time

Environmental harm level

Recycling & disposal tips

Upcycle ideas

Gamified eco-points

Eco streaks & badges

A personal dashboard

AI-powered eco chat

EcoSnap promotes practical environmental awareness through fast AI detection and fun gamification 🌍💚.

# ✨ Features
### 🧪 1. Smart AI Waste Detection

✔ Upload photo
✔ Capture via live camera
✔ 224×224 CNN model (trained on merged dataset)
✔ Instant classification
✔ Confidence score

### 🌱 2. Environmental Knowledge Lookup

For every detected item, the app shows:

🏷 Class Name
⏳ Decomposition Duration
⚠ Harm Level
♻ Recycling Tips
🎨 Upcycle Ideas

### 🤖 3. Eco Chat Assistant (Groq AI)

Ask anything related to:

Recycling

Waste categories

Climate impact

Eco-friendly lifestyle

Uses Groq LLaMA-3.3-70B for ultra-fast responses.

### 🎮 4. Gamification

Earn points for every scan!

Achievement	Unlock Condition
🥇 First Scan	Complete your first detection
🌱 Eco Beginner	5 scans
♻ Recycler	15 scans
🌍 Green Hero	50 scans

Also includes:

Daily streak tracking

Level progression

Scan history

### 📊 5. User Dashboard

Includes:

Total eco-points

Scans today

Total scans

Current level

Daily streak

Recent 10 scans

Achievements

#### 📂 Project Structure
ecosnap-app/
│
├── app.py                     # Main Streamlit app
├── eco_database.json          # Waste impact information
├── upcycle_ideas.json         # Upcycle suggestions
├── user_data.json             # Gamification + Stats
├── label_map_merged.json      # Label mapping for model
├── waste_model_merged.keras   # Trained TensorFlow model
│
├── dataset-resized/           # (optional) cleaned dataset
├── merged_dataset/            # final merged dataset
├── taco_classes/              # TACO dataset classes
│
├── .streamlit/
│    └── secrets.toml          # API key storage
│
└── README.md

### 🛠 Installation Guide
1. Clone Repository
git clone https://github.com/YOUR_USERNAME/ecosnap-app.git
cd ecosnap-app

2. Install Requirements
pip install -r requirements.txt

3. Add Groq API Key

Create folder:

.streamlit/secrets.toml


Inside:

GROQ_API_KEY="your_api_key_here"


4. Run the App
streamlit run app.py



### 🧠 Model Information
Property	Details
Framework	TensorFlow / Keras
Input Size	224×224
Dataset	TACO + Custom Waste Dataset
Output	Softmax classification
Model File	waste_model_merged.keras

### 💾 Datasets

You can upload your dataset repo separately. Recommended structure:

datasets/
├── plastic/
├── paper/
├── cardboard/
├── metal/
└── glass/


A separate dataset repo README can also be generated upon request.

### 🖼 Screenshots (Add your real screenshots later)
🏠 Home Page

🔍 Detection Page

📸 Camera Capture

📊 Dashboard

🤖 Eco Chat


### 🌍 Why EcoSnap Matters

Waste mismanagement harms our:

Oceans

Soil fertility

Air quality

Wildlife

EcoSnap helps people:

Learn environmental impact

Improve waste sorting habits

Reduce landfill waste

Adopt eco-friendly lifestyles

Technology for a better future 🌎💚


## 👨‍💻 Developer

Vineet Kumar
B.Tech CSE – GLA University
AI • Sustainability • Full-Stack Development

