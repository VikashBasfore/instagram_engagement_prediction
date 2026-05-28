print("Training Started...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv("instagram_usage_lifestyle.xls")
df = data.copy()


# not necessary 
df.drop('last_login_date', axis=1, inplace=True)

# The 'app_name' column contained only a single value (Instagram)
df.drop('app_name', axis=1, inplace=True)

# Drop unnecessary columns 
df = df.drop(['account_creation_year'], axis=1)

# Feature Enginering
# Feature Enginering
# 1. Sleep_hours_per_night
def sleep_category(x):
    if x < 6:
        return "Low"
    elif x <= 8:
        return "Normal"
    else:
        return "High"

df['sleep_cat'] = df['sleep_hours_per_night'].apply(sleep_category)


# 2. BMI
def bmi_category(x):
    if x < 18.5:
        return "Underweight"
    elif x < 25:
        return "Normal"
    elif x < 30:
        return "Overweight"
    else:
        return "Obese"

df['bmi_cat'] = df['body_mass_index'].apply(bmi_category)


# 4. Health Index
# Combines happiness (positive) and stress (negative)
# Helps model understand actual mental condition instead of separate signals
df['health_index'] = df['self_reported_happiness'] - df['perceived_stress_score']


# 5. Activity Ratio
# Compares physical activity (steps) vs sedentary behavior (Instagram usage)
# High value = active lifestyle, Low value = inactive/addictive behavior
df['activity_ratio'] = df['daily_steps_count'] / (df['daily_active_minutes_instagram'] + 1)


# 6. Follower Ratio
# Measures influence instead of raw followers
# High ratio = influencer-type user, Low ratio = normal user
df['follower_ratio'] = df['followers_count'] / (df['following_count'] + 1)


# 7 Blood Pressure Feature
# Blood Pressure Category (important medical feature)
def bp_category(x):
    if x < 120:
        return "Normal"
    elif x < 130:
        return "Elevated"
    elif x < 140:
        return "High_Stage_1"
    else:
        return "High_Stage_2"

df['bp_cat'] = df['blood_pressure_systolic'].apply(bp_category)
df['activity_per_day'] = df['daily_steps_count'] + df['social_events_per_month']
df['social_score'] = df['dms_sent_per_week'] + df['dms_received_per_week']
df['social_activity'] = df['dms_sent_per_week'] + df['dms_received_per_week']
df['life_balance'] = df['hobbies_count'] + df['social_events_per_month']
df['activity_efficiency'] = df['daily_steps_count'] / (df['sleep_hours_per_night'] + 1)




# Fix highly skewed features

skewed_cols = [
    'followers_count',
    'activity_ratio',
    'follower_ratio',
    'sessions_per_day'
]

for col in skewed_cols:
    df[col] = np.log1p(df[col])
    
# target variable
# Convert target into 3 balanced categories
def engagement_category(x):
    if x < 1.5:
        return 'Low'
    elif x < 3:
        return 'Medium'
    else:
        return 'High'

df['engagement_level'] = df['user_engagement_score'].apply(engagement_category)

df['engagement_level'] = df['engagement_level'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})

df = df.drop(['user_engagement_score'], axis=1)



# ENCODING


# Gender 
df = pd.get_dummies(df, columns=['gender'], drop_first=True)
# country
df = pd.get_dummies(df, columns=['country'], drop_first=True)
# urban_rural
df = pd.get_dummies(df, columns=['urban_rural'], drop_first=True)
# income_level (ordinal encording)
df['income_level'] = df['income_level'].map({
    'Low': 0,
    'Lower-middle': 1,
    'Middle': 2,
    'Upper-middle': 3,
    'High': 4
})
# employment_status
df = pd.get_dummies(df, columns=['employment_status'], drop_first=True)
# education_level
df['education_level'] = df['education_level'].map({
    'High school': 0,
    'Some college': 1,
    'Bachelor’s': 2,
    'Master’s': 3,
    'PhD': 4,
    'Other': 2   # treated as mid-level (safe assumption)
})

# relationship_status
df = pd.get_dummies(df, columns=['relationship_status'], drop_first=True)

# has_children (Binary Encoding)
df['has_children'] = df['has_children'].map({
    'No': 0,
    'Yes': 1
})

# diet_quality (ordinal encoding)
df['diet_quality'] = df['diet_quality'].map({
    'Very poor': 0,
    'Poor': 1,
    'Average': 2,
    'Good': 3,
    'Excellent': 4
})

# smoking (Ordinal Encoding)
df['smoking'] = df['smoking'].map({
    'No': 0,
    'Former': 1,
    'Yes': 2
})

# alcohol_frequency
df['alcohol_frequency'] = df['alcohol_frequency'].map({
    'Never': 0,
    'Rarely': 1,
    'Weekly': 2,
    'Several times a week': 3,
    'Daily': 4
})

# uses_premium_features
df['uses_premium_features'] = df['uses_premium_features'].map({
    'No': 0,
    'Yes': 1
})

# content_type_preference
df = pd.get_dummies(df, columns=['content_type_preference'], drop_first=True)

# preferred_content_theme
df = pd.get_dummies(df, columns=['preferred_content_theme'], drop_first=True)

# privacy_setting_level
df = pd.get_dummies(df, columns=['privacy_setting_level'], drop_first=True)

# two_factor_auth_enabled
df['two_factor_auth_enabled'] = df['two_factor_auth_enabled'].map({
    'No': 0,
    'Yes': 1
})

# biometric_login_used
df['biometric_login_used'] = df['biometric_login_used'].map({
    'No': 0,
    'Yes': 1
})

# subscription_status
df = pd.get_dummies(df, columns=['subscription_status'], drop_first=True)

# sleep_cat
df = pd.get_dummies(df, columns=['sleep_cat'], drop_first=True)

# bmi_cat
df = pd.get_dummies(df, columns=['bmi_cat'], drop_first=True)

# bp_cat
df = pd.get_dummies(df, columns=['bp_cat'], drop_first=True)

# 1. Convert bool → int
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# 2. leakage columns
leakage_cols = [
   'likes_given_per_day',
   'comments_written_per_day',
    # 'engagement_intensity',
    # 'content_consumption',
    'time_on_feed_per_day',
    'time_on_reels_per_day',
    'sessions_per_day'
]

df = df.drop(columns=leakage_cols)


# Model
from sklearn.model_selection import train_test_split

# Define X and y 
X = df.drop('engagement_level', axis=1) 
y = df['engagement_level']

# Split 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# Model 1

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# =========================
# MODEL
# =========================
model_l = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    n_jobs=-1,
    random_state=42
)

# =========================
# TRAIN
# =========================
model_l.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = model_l.predict(X_test)

# =========================
# ACCURACY
# =========================
print("Accuracy:", accuracy_score(y_test, y_pred))

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# =========================
# CONFUSION MATRIX PLOT
# =========================
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# =========================
# TRAIN VS TEST SCORE
# (CHECK OVERFITTING)
# =========================
train_score = model_l.score(X_train, y_train)
test_score = model_l.score(X_test, y_test)

print("\nTrain Accuracy :", train_score)
print("Test Accuracy  :", test_score)


# =========================
# SAVE MODEL
# =========================

import joblib

joblib.dump(model_l, "model.pkl")

joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("\nModel Saved Successfully")

print('Training End.....')