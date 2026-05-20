# app.py
import streamlit as st
import pandas as pd
import pickle
import os
import subprocess

# Set up clean, professional page configurations
st.set_page_config(
    page_title="Iris Flower Classifier", 
    page_icon="🌸", 
    layout="centered"
)

st.title("🌸 Iris Flower Species Classifier")
st.write(
    "This web application uses a Machine Learning model (Random Forest) "
    "to predict the species of an Iris flower based on its structural dimensions."
)

# Set path targets
model_path = os.path.join('models', 'iris_model.pkl')
script_path = os.path.join('src', 'train.py')

# Create a clear sidebar section for system operations
st.sidebar.header("⚙️ Model Management")

# Retrain Button Logic with explicit environment pathing for Windows stability
if st.sidebar.button("🔄 Retrain Model Pipeline"):
    with st.spinner("Running training pipeline... Please wait."):
        
        # Windows Workaround: Point directly to the python execution engine in your local venv
        venv_python = os.path.join('venv', 'Scripts', 'python.exe')
        
        # Run your background script safely through python
        result = subprocess.run([venv_python, script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            st.sidebar.success("✅ Model retrained successfully!")
        else:
            st.sidebar.error("❌ Error running training script:")
            st.sidebar.code(result.stderr)

st.sidebar.markdown("---")
st.sidebar.header("📐 Input Flower Dimensions")

# Verify if a model exists before building the slider user interface
if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
    st.warning(
        "⚠️ No trained model found! Please click 'Retrain Model Pipeline' "
        "in the sidebar or run 'python src/train.py' in your terminal to initialize."
    )
else:
    # Safely load the model binary file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # User sliders configuration 
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, step=0.1)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.4, step=0.1)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.5, step=0.1)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.4, step=0.1)

    # Structure user inputs into a DataFrame format matching features from 'Iris.csv'
    input_data = pd.DataFrame([{
        'SepalLengthCm': sepal_length,
        'SepalWidthCm': sepal_width,
        'PetalLengthCm': petal_length,
        'PetalWidthCm': petal_width
    }])

    # Display user inputs dynamically on the web panel
    st.subheader("Your Input Parameters")
    st.dataframe(input_data)

    # Classification computation step
    if st.button("Predict Species", type="primary"):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)
        
        st.markdown("---")
        # Output prediction result explicitly
        st.success(f"🎉 The model predicts this flower belongs to the species: **{prediction}**")
        
        # Visualize prediction confidence levels using horizontal bar graph charts
        st.subheader("📊 Prediction Confidence / Probability Distribution")
        prob_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.bar_chart(prob_df.T)