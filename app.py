import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import streamlit as slt
from sklearn.ensemble import RandomForestClassifier
import joblib as jb
## import model/dataframe with the joblib

df_filtered = pd.read_csv("Filtered_Shipment_Date.csv")

model = jb.load("dt_model.pkl")

model_columns = jb.load("model_columns.pkl")

## begin setup of front page

slt.set_page_config(page_title="Project Title", layout="centered")
slt.title("Supply Chain Delay Predictor")
slt.write("Enter the shipment details below to predict if the package will arrive late")

slt.subheader("Shipment Details")

# creates split column + dropdown from df

col1, col2 = slt.columns(2)
with col1:
    selected_vendor = slt.selectbox("Select Vendor", df_filtered['vendor'].unique())
    selected_country = slt.selectbox("Destination Country", df_filtered['country'].unique())
with col2:
    selected_mode = slt.selectbox("Shipment Mode", df_filtered['shipment mode'].dropna().unique())
    selected_fulfill = slt.selectbox("Fulfill Via", df_filtered['fulfill via'].unique())

## check

if slt.button("Run Prediction", type="primary"):
    
    # Format the input data
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0 
    
    if f"vendor_{selected_vendor}" in input_data.columns:
        input_data[f"vendor_{selected_vendor}"] = 1
    if f"country_{selected_country}" in input_data.columns:
        input_data[f"country_{selected_country}"] = 1
    if f"shipment mode_{selected_mode}" in input_data.columns:
        input_data[f"shipment mode_{selected_mode}"] = 1
    if f"fulfill via_{selected_fulfill}" in input_data.columns:
        input_data[f"fulfill via_{selected_fulfill}"] = 1


    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

## Results
    slt.divider()
    slt.subheader("Prediction Results")

    if prediction == "True":
        slt.error(f"**HIGH RISK OF DELAY**")
        slt.write(f"The model is **{probability * 100:.1f}%** confident this shipment will be late.")
    else:
        slt.success(f"**LIKELY ON TIME**")
        slt.write(f"The model is **{(1 - probability) * 100:.1f}%** confident this shipment will arrive on schedule.")


## SHAP MAGIC TIME

    slt.divider()
    slt.subheader("Why did AI make this prediction?")

## calculate shap for user input
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)


# Safely extract SHAP values regardless of library version
    if isinstance(shap_values, list):
        # List format: [class_0, class_1]
        instance_shap = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
        # 3D array format: (samples, features, classes)
        instance_shap = shap_values[0, :, 1]
    else:
        # 2D array format: (samples, features)
        instance_shap = shap_values[0]

# Isolate the top 5 features that pushed the prediction the hardest
    top_indices = np.argsort(np.abs(instance_shap))[-5:]
    top_features = input_data.columns[top_indices]
    top_values = instance_shap[top_indices]

# plot it
    fig, axes = plt.subplots(figsize=(8,4))

# Color code: pinkadink if it pushes towards Late, green if it pushes towards On Time
    colors = ['#FF1493' if val > 0 else "#46c65e" for val in top_values]

    axes.set_title("Top 5 factors for this specific shipment")
    axes.barh(top_features, top_values, color=colors)
    axes.set_xlabel("Impact on Lateness(SHAP Value)")


    slt.pyplot(fig)
    
    slt.caption("Pink shows model was pushed to guess 'Late', green represents 'On Time'.")

