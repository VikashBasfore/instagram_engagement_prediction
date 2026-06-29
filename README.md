# 📱 Instagram Engagement AI Predictor

<div align="center">

### 🚀 AI-Powered Instagram Engagement Prediction using Machine Learning, Streamlit & Google Gemini

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![Google Gemini](https://img.shields.io/badge/Google-Gemini-blue)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Predict Instagram user engagement using Machine Learning, advanced feature engineering, lifestyle analytics, and an AI-powered recommendation assistant.

</div>

---

# 📌 Project Overview

Instagram engagement depends on multiple lifestyle, behavioral, and social media factors. Instead of relying only on follower count or likes, this project combines health metrics, activity patterns, lifestyle habits, and Instagram usage statistics to predict whether a user's engagement level is **Low**, **Medium**, or **High**.

The project also includes a modern Streamlit dashboard and an AI assistant powered by **Google Gemini** that provides personalized creator growth recommendations.

---

# 🎯 Objectives

* Predict Instagram engagement level.
* Analyze user lifestyle and social media behavior.
* Build a production-ready Machine Learning model.
* Deploy an interactive Streamlit dashboard.
* Generate personalized AI-powered creator advice.

---

# 📂 Dataset

The dataset contains Instagram usage statistics together with user lifestyle and health information.

### Features Include

* User demographics
* Followers & Following
* Instagram activity
* Reels and Stories usage
* Physical activity
* Sleep patterns
* BMI
* Blood pressure
* Diet quality
* Stress and happiness scores
* Social interactions
* Privacy settings
* Premium subscription status

Target Variable:

**Engagement Level**

* 🔴 Low
* 🟡 Medium
* 🟢 High

---

# 🛠 Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* LightGBM
* Scikit-learn

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Deployment

* Streamlit

### AI Integration

* Google Gemini API

### Model Storage

* Joblib

---

# ⚙️ Machine Learning Pipeline

```text
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Encoding
    │
    ▼
Log Transformation
    │
    ▼
Train-Test Split
    │
    ▼
LightGBM Training
    │
    ▼
Model Evaluation
    │
    ▼
Model Saving
    │
    ▼
Streamlit Deployment
    │
    ▼
Google Gemini AI Assistant
```

---

# 🧠 Feature Engineering

Several custom features were created to improve predictive performance.

### Lifestyle Features

* Sleep Category
* BMI Category
* Blood Pressure Category
* Health Index
* Activity Ratio
* Activity Efficiency

### Social Features

* Follower Ratio
* Social Score
* Social Activity
* Life Balance

### Data Transformations

* Log transformation for highly skewed variables
* One-Hot Encoding
* Ordinal Encoding
* Binary Encoding
* Leakage feature removal

---

# 🤖 Machine Learning Model

Model Used:

✅ LightGBM Classifier

Configuration

* 300 Estimators
* Learning Rate = 0.05
* Max Depth = 8
* Num Leaves = 31

Evaluation Metrics

* Accuracy
* Classification Report
* Confusion Matrix
* Train Accuracy
* Test Accuracy

---

# 🌐 Streamlit Dashboard

The project includes an interactive web application where users can:

* Adjust creator profile information.
* Predict engagement level instantly.
* View engagement probabilities.
* Explore interactive charts.
* Analyze creator metrics.
* Receive AI-powered growth suggestions using Google Gemini.

---

# 📊 Dashboard Features

* 📈 Engagement Probability Chart
* 📊 Creator Analytics Cards
* 📉 Radar Chart
* 🤖 AI Prediction Panel
* 💡 Personalized AI Creator Assistant
* 🌙 Modern Dark Theme UI

---

# 📁 Repository Structure

```text
Instagram-Engagement-AI
│
├── app.py
├── model.pkl
├── model_columns.pkl
├── style.css
├── requirements.txt
├── instagram_usage_lifestyle.csv
├── notebook.ipynb
├── README.md
└── images/
```

---

# 🚀 Installation

Clone Repository

```bash
git clone https://github.com/yourusername/Instagram-Engagement-AI.git
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run Application

```bash
streamlit run app.py
```

---

# 💻 AI Features

The application integrates Google Gemini to provide:

* Personalized Instagram growth strategies.
* Engagement improvement recommendations.
* Creator performance insights.
* AI-generated answers to user questions.

---

# 📈 Key Features

* End-to-End Machine Learning Pipeline
* Advanced Feature Engineering
* LightGBM Classification Model
* Interactive Streamlit Dashboard
* Plotly Visualizations
* AI Creator Assistant
* Real-time Engagement Prediction
* Professional Dark Theme Interface

---

# 🔮 Future Improvements

* Deep Learning Models
* SHAP Explainability
* User Authentication
* Cloud Deployment
* Instagram API Integration
* Automated Retraining Pipeline

---

# 🙏 Acknowledgements

Special thanks to the open-source community and the developers of:

* Streamlit
* Scikit-learn
* LightGBM
* Plotly
* Google Gemini API
* Pandas
* NumPy

Their tools made the development of this project possible.

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a Star!

</div>
