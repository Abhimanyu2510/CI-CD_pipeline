import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Clustering App", layout="wide")

def train_model(data, n_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(scaled_data)
    
    # Save models
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return model, scaler, scaled_data

def predict(data, model, scaler):
    scaled_data = scaler.transform(data)
    predictions = model.predict(scaled_data)
    return predictions

def main():
    st.title("Clustering Application")
    
    tab1, tab2 = st.tabs(["Training", "Inference"])
    
    with tab1:
        st.header("Model Training")
        uploaded_file = st.file_uploader("Upload training data (CSV)", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data.head())
            
            feature_cols = st.multiselect("Select features for clustering", data.columns)
            n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
            
            if st.button("Train Model") and feature_cols:
                training_data = data[feature_cols]
                model, scaler, scaled_data = train_model(training_data, n_clusters)
                
                # Visualize results
                predictions = model.predict(scaled_data)
                data['Cluster'] = predictions
                
                if len(feature_cols) >= 2:
                    fig = px.scatter(
                        data, 
                        x=feature_cols[0],
                        y=feature_cols[1],
                        color='Cluster',
                        title='Clustering Results'
                    )
                    st.plotly_chart(fig)
                
                st.success("Model trained successfully!")
    
    with tab2:
        st.header("Model Inference")
        inference_file = st.file_uploader("Upload data for inference (CSV)", type="csv", key="inference")
        
        if inference_file is not None and os.path.exists('model.joblib'):
            inference_data = pd.read_csv(inference_file)
            st.dataframe(inference_data.head())
            
            model = joblib.load('model.joblib')
            scaler = joblib.load('scaler.joblib')
            
            if st.button("Run Inference"):
                predictions = predict(inference_data, model, scaler)
                inference_data['Predicted_Cluster'] = predictions
                
                st.write("Predictions:")
                st.dataframe(inference_data)
                
                # Download predictions
                st.download_button(
                    label="Download predictions",
                    data=inference_data.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Please train a model first!")

if __name__ == "__main__":
    main()
