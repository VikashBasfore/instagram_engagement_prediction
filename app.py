import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai

from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)
# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Instagram Engagement AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD CSS
# ==========================================

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚙ Creator Controls")

age = st.sidebar.slider(
    "Age",
    18,
    100,
    25
)

followers_count = st.sidebar.number_input(
    "Followers Count",
    min_value=0,
    value=10000,
    step=100
)

following_count = st.sidebar.number_input(
    "Following Count",
    min_value=0,
    value=10000,
    step=2000
)

posts_created_per_week = st.sidebar.slider(
    "Posts Created Per Week",
    0,
    100,
    10
)

daily_active_minutes_instagram = st.sidebar.slider(
    "Instagram Minutes",
    0,
    600,
    180
)

stories_viewed_per_day = st.sidebar.slider(
    "Stories Viewed Per Day",
    0,
    500,
    100
)

reels_watched_per_day = st.sidebar.slider(
    "Reels Watched Per Day",
    0,
    500,
    150
)

likes_given_per_day = st.sidebar.slider(
    "Likes Given Per Day",
    0,
    1000,
    250
)

comments_written_per_day = st.sidebar.slider(
    "Comments Written Per Day",
    0,
    200,
    30
)

exercise_hours_per_week = st.sidebar.slider(
    "Exercise Hours Per Week",
    0.0,
    24.0,
    5.0
)

average_session_length_minutes = st.sidebar.slider(
    "Average Session Length",
    1,
    180,
    45
)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

activity_ratio = (
    daily_active_minutes_instagram /
    (exercise_hours_per_week + 1)
)

follower_ratio = (
    followers_count /
    (following_count + 1)
)

content_consumption = (
    reels_watched_per_day +
    stories_viewed_per_day
)

engagement_intensity = (
    likes_given_per_day +
    comments_written_per_day
)

# ==========================================
# INPUT DATAFRAME
# ==========================================

input_data = {
    'age': age,
    'followers_count': followers_count,
    'following_count': following_count,
    'posts_created_per_week': posts_created_per_week,
    'daily_active_minutes_instagram': daily_active_minutes_instagram,
    'stories_viewed_per_day': stories_viewed_per_day,
    'reels_watched_per_day': reels_watched_per_day,
    'likes_given_per_day': likes_given_per_day,
    'comments_written_per_day': comments_written_per_day,
    'exercise_hours_per_week': exercise_hours_per_week,
    'average_session_length_minutes': average_session_length_minutes,
    'activity_ratio': activity_ratio,
    'follower_ratio': follower_ratio,
    'content_consumption': content_consumption,
    'engagement_intensity': engagement_intensity
}

input_df = pd.DataFrame([input_data])

# ==========================================
# MATCH MODEL COLUMNS
# ==========================================

for col in model_columns:

    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_columns]

# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero-box">

<div class="hero-title">
📱 Instagram Engagement AI
</div>

<div class="hero-subtitle">
Futuristic Social Media Analytics Dashboard
</div>

</div>
""", unsafe_allow_html=True)

# ==========================================
# KPI BOXES
# ==========================================

st.markdown(f"""

<div class="kpi-container">

<div class="kpi-card">
<div class="kpi-icon">👥</div>
<div class="kpi-title">Followers</div>
<div class="kpi-value">Strong Growth</div>
<div class="kpi-glow"></div>
</div>

<div class="kpi-card">
<div class="kpi-icon">📝</div>
<div class="kpi-title">Posts / Week</div>
<div class="kpi-value">Daily Active</div>
<div class="kpi-glow"></div>
</div>

<div class="kpi-card">
<div class="kpi-icon">🔥</div>
<div class="kpi-title">Engagement</div>
<div class="kpi-value">High Reach</div>
<div class="kpi-glow"></div>
</div>

<div class="kpi-card">
<div class="kpi-icon">⏱</div>
<div class="kpi-title">Session Length</div>
<div class="kpi-value">Peak Retention</div>
<div class="kpi-glow"></div>
</div>

</div>
""", unsafe_allow_html=True)



# ==========================================
# SESSION STATE
# ==========================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
# ==========================================
# PREDICT BUTTON
# ==========================================

predict = st.button(
    "🚀 Analyze Creator Profile"
)

if predict:
    st.session_state.prediction_done = True

# ==========================================
# PREDICTION
# ==========================================

if st.session_state.prediction_done:

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    # ======================================
    # RESULT LOGIC
    # ======================================

    if prediction == 0:

        result = "LOW ENGAGEMENT"
        color = "#ff006e"

        insight = """
        Audience interaction is weak.
        Reels and content engagement are below average.
        """

    elif prediction == 1:

        result = "MEDIUM ENGAGEMENT"
        color = "#ffd60a"

        insight = """
        Balanced engagement with moderate audience activity.
        """

    else:

        result = "HIGH ENGAGEMENT"
        color = "#00f5d4"

        insight = """
        Strong creator activity with excellent audience interaction.
        """

    # ======================================
    # RESULT BOX
    # ======================================

    st.markdown(f"""
    <div class="anime-prediction-card">

    <div class="anime-character">

    <img src="https://cdn-icons-png.flaticon.com/512/4140/4140048.png"
    class="anime-img">
    </div>

    <div class="anime-text-section">

    <div class="ai-badge">
        ⚡ AI Prediction
    </div>

    <div class="anime-result">
        {result}
    </div>

    <div class="anime-description">
        {insight}
    </div>

    </div>

    </div>
    """, unsafe_allow_html=True)

    # ======================================
    # CHARTS
    # ======================================

    chart1, chart2 = st.columns(2)

    with chart1:

        prob_df = pd.DataFrame({
            'Engagement': ['Low', 'Medium', 'High'],
            'Probability': probabilities
        })

        fig = px.bar(
            prob_df,
            x='Engagement',
            y='Probability',
            color='Engagement',
            text='Probability',
            template='plotly_dark'
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with chart2:

        radar = go.Figure()

        radar.add_trace(go.Scatterpolar(
            r=[
                posts_created_per_week,
                reels_watched_per_day,
                stories_viewed_per_day,
                likes_given_per_day,
                comments_written_per_day
            ],

            theta=[
                'Posts',
                'Reels',
                'Stories',
                'Likes',
                'Comments'
            ],

            fill='toself'
        ))

        radar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )

        st.plotly_chart(
            radar,
            use_container_width=True
        )

    # ======================================
    # ANALYTICS CARDS
    # ======================================

    st.markdown("## 📊 Creator Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class='analytics-card'>

        <h3>🔥 Engagement Intensity</h3>

        <h1>{engagement_intensity}</h1>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class='analytics-card'>

        <h3>📈 Follower Ratio</h3>

        <h1>{round(follower_ratio, 2)}</h1>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class='analytics-card'>

        <h3>🎥 Content Consumption</h3>

        <h1>{content_consumption}</h1>

        </div>
        """, unsafe_allow_html=True)

    # ======================================
    # AI INSIGHTS
    # ======================================

    st.markdown("## 🤖 AI Insights")

    st.markdown(f"""
    <div class='insight-box'>

    ✅ Audience engagement patterns analyzed successfully.<br><br>

    ✅ Creator activity metrics indicate platform consistency.<br><br>

    ✅ AI model powered by LightGBM analytics.<br><br>

    ✅ Reels and interaction behavior strongly influence engagement.

    </div>
    """, unsafe_allow_html=True)


    # ======================================
    # AI ASSISTANT
    # ======================================

    st.markdown("## 🤖 AI Creator Assistant")

    user_question = st.text_input(
        "Ask AI about Instagram growth"
    )

    if st.button("✨ Generate AI Advice"):

        prompt = f"""
        You are an Instagram growth expert.

        Engagement Prediction: {result}

        User Metrics:
        Followers: {followers_count}
        Following: {following_count}
        Posts per week: {posts_created_per_week}
        Reels watched: {reels_watched_per_day}
        Likes given: {likes_given_per_day}

        User Question:
        {user_question}

        Give professional creator advice.
        """

        with st.spinner("🤖 AI is generating creator strategy..."):

            response = gemini_model.generate_content(prompt)

        st.markdown(
            '<div class="assistant-card">',
            unsafe_allow_html=True
        )

        st.write(response.text)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;
            color:gray;'>

Built with Streamlit • LightGBM • AI Analytics

</div>
""", unsafe_allow_html=True)

