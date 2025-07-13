import streamlit as st

st.set_page_config(page_title="Teen Phone Addiction Analyzer", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore")

@st.cache_data
def load_data():
    if not os.path.exists("teen_phone_addiction_dataset.csv"):
        st.error("❌ CSV file not found. Please upload 'teen_phone_addiction_dataset.csv'.")
        st.stop()
    df = pd.read_csv("teen_phone_addiction_dataset.csv")
    df["Addiction_Category"] = pd.cut(df["Addiction_Level"], bins=[0, 6, 8, 10], labels=["Low", "Moderate", "High"])
    cat_cols = ["Gender", "Location", "School_Grade", "Phone_Usage_Purpose"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

df, label_encoders = load_data()

features = [
    'Age', 'Gender', 'Location', 'School_Grade',
    'Daily_Usage_Hours', 'Sleep_Hours', 'Academic_Performance', 'Social_Interactions',
    'Exercise_Hours', 'Anxiety_Level', 'Depression_Level', 'Self_Esteem',
    'Parental_Control', 'Screen_Time_Before_Bed', 'Phone_Checks_Per_Day',
    'Apps_Used_Daily', 'Time_on_Social_Media', 'Time_on_Gaming',
    'Time_on_Education', 'Phone_Usage_Purpose', 'Family_Communication', 'Weekend_Usage_Hours'
]
X = df[features]
y = df["Addiction_Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

st.title("📱 Teen Phone Addiction Risk Analyzer")
menu = st.sidebar.radio("Navigation", ["Dashboard", "Visualization", "Predict Addiction"])

if menu == "Dashboard":
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(10))
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Records", df.shape[0])
        st.metric("Model Accuracy", f"{model.score(X_test, y_test) * 100:.2f}%")
    with col2:
        st.bar_chart(df["Addiction_Category"].value_counts())
    st.subheader("📌 Statistical Summary")
    st.write(df.describe())

elif menu == "Visualization":
    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[features + ["Addiction_Level"]].corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Scatter Plot")
    x_axis = st.selectbox("X-axis", features)
    y_axis = st.selectbox("Y-axis", ["Addiction_Level", "Daily_Usage_Hours", "Sleep_Hours"])
    color = st.selectbox("Color By", ["Addiction_Category", "Gender", "Phone_Usage_Purpose"])

    fig2 = px.scatter(df, x=x_axis, y=y_axis, color=color, size="Phone_Checks_Per_Day", hover_data=["Age"])
    st.plotly_chart(fig2)

elif menu == "Predict Addiction":
    st.subheader("🔮 Predict Teen Addiction Risk")
    user_input = {}
    for feat in features:
        if feat in label_encoders:
            options = label_encoders[feat].classes_.tolist()
            user_input[feat] = label_encoders[feat].transform([st.selectbox(feat, options)])[0]
        elif df[feat].dtype == "int64":
            user_input[feat] = st.slider(feat, int(df[feat].min()), int(df[feat].max()), int(df[feat].mean()))
        else:
            user_input[feat] = st.slider(feat, float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"📱 Predicted Addiction Category: **{prediction}**")
        if prediction == "High":
            st.warning("⚠️ High Risk: Recommend reducing screen time and increasing physical/social activity.")
        elif prediction == "Moderate":
            st.info("🟡 Moderate Risk: Monitor and guide screen habits.")
        else:
            st.success("🟢 Low Risk: Healthy usage pattern detected.")