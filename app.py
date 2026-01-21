import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="Renewable Energy Prediction",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Renewable Energy Output Prediction System")
st.markdown(
    "This application predicts **renewable energy output** using **weather parameters** "
    "and a **machine learning model (Random Forest)**."
)
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset.csv")
    df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)
    return df

df = load_data()

features = [
    "Pressure",
    "global_radiation",
    "temp_mean(c)",
    "temp_min(c)",
    "temp_max(c)",
    "Wind_Speed",
    "Wind_Bearing"
]

X = df[features]
y = df["normalized_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.sidebar.header("🔧 Weather Parameters")

pressure = st.sidebar.slider("Pressure", 900, 1100, 1013)
global_radiation = st.sidebar.slider("Global Radiation", 0, 1000, 300)
temp_mean = st.sidebar.slider("Mean Temperature (°C)", -10, 50, 25)
temp_min = st.sidebar.slider("Min Temperature (°C)", -20, 40, 18)
temp_max = st.sidebar.slider("Max Temperature (°C)", -10, 60, 32)
wind_speed = st.sidebar.slider("Wind Speed", 0, 30, 5)
wind_bearing = st.sidebar.slider("Wind Bearing", 0, 360, 180)

st.subheader("📊 Prediction Result")

if st.button("🔮 Predict Energy Output"):
    user_data = [[
        pressure,
        global_radiation,
        temp_mean,
        temp_min,
        temp_max,
        wind_speed,
        wind_bearing
    ]]
    
    user_data_scaled = scaler.transform(user_data)
    prediction = rf_model.predict(user_data_scaled)

    st.success(f"⚡ Predicted Renewable Energy Output: **{prediction[0]:.4f}**")

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)
col1.metric("R² Score", round(r2, 3))
col2.metric("Mean Absolute Error", round(mae, 4))
st.subheader("📉 Visual Analysis")

tab1, tab2 = st.tabs(["Correlation Heatmap", "Actual vs Predicted"])

with tab1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        df[features + ["normalized_label"]].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred)
    ax2.set_xlabel("Actual Energy Output")
    ax2.set_ylabel("Predicted Energy Output")
    ax2.set_title("Actual vs Predicted Energy Output")
    st.pyplot(fig2)

st.markdown("---")
st.markdown("Developed using **Machine Learning & Streamlit** 🚀")
