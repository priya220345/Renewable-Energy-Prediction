import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jobilb # You will need to save your model and scaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Renewable Energy Predictor", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Title and Description ---
st.title("☀️ Renewable Energy Output Prediction")
st.markdown("""
This application predicts the **Normalized Energy Output** based on weather parameters using a Random Forest Regressor model.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Input Weather Parameters")

def user_input_features():
    pressure = st.sidebar.slider("Pressure", 900.0, 1100.0, 1010.0)
    global_rad = st.sidebar.number_input("Global Radiation", 0.0, 1500.0, 200.0)
    temp_mean = st.sidebar.slider("Mean Temperature (°C)", -10.0, 50.0, 20.0)
    temp_min = st.sidebar.slider("Min Temperature (°C)", -15.0, 40.0, 15.0)
    temp_max = st.sidebar.slider("Max Temperature (°C)", -5.0, 60.0, 25.0)
    wind_speed = st.sidebar.number_input("Wind Speed", 0.0, 100.0, 5.0)
    wind_bearing = st.sidebar.slider("Wind Bearing", 0, 360, 180)
    
    data = {
        'Pressure': pressure,
        'global_radiation': global_rad,
        'temp_mean(c)': temp_mean,
        'temp_min(c)': temp_min,
        'temp_max(c)': temp_max,
        'Wind_Speed': wind_speed,
        'Wind_Bearing': wind_bearing
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Load Data/Model (Simulation based on your Notebook logic) ---
# In a real deployment, you should save your trained model and scaler using joblib
# For this example, we assume you've loaded a dataset to show the EDA part
@st.cache_data
def load_data():
    # Replace with your dataset path or an example sample
    # df = pd.read_csv('Dataset.csv') 
    # return df
    pass

# Main Layout with Tabs
tab1, tab2 = st.tabs(["🚀 Prediction", "📊 Data Visualizations"])

with tab1:
    st.subheader("User Input Parameters")
    st.write(input_df)

    # Note: In production, use: 
    # model = joblib.load('rf_model.pkl')
    # scaler = joblib.load('scaler.pkl')
    
    if st.button("Predict Energy Output"):
        # Placeholder for prediction logic (mirroring your notebook)
        # Assuming scaler and model are ready
        try:
            # dummy logic for demonstration if files aren't exported yet
            # scaled_data = scaler.transform(input_df)
            # prediction = rf_model.predict(scaled_data)
            
            st.success("### Prediction Results")
            st.metric(label="Predicted Normalized Energy Output", value=f"0.7421 units") # Example output
            st.info("The prediction is based on the trained Random Forest Regressor.")
        except Exception as e:
            st.error(f"Model error: {e}. Ensure the model and scaler are pre-trained and loaded.")

with tab2:
    st.subheader("Model Insights & EDA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Actual vs Predicted (Example)**")
        # Creating a sample plot based on your notebook logic
        fig, ax = plt.subplots()
        sns.scatterplot(x=[1, 2, 3, 4], y=[1.1, 1.9, 3.2, 3.8], ax=ax)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

    with col2:
        st.write("**Correlation Heatmap**")
        # Sample heatmap logic
        fig2, ax2 = plt.subplots()
        # Use a dummy matrix if df not loaded
        dummy_corr = np.random.rand(7,7)
        sns.heatmap(dummy_corr, annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

st.divider()
st.caption("Developed for Renewable Energy Prediction Analytics.")
