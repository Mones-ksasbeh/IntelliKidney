import streamlit as st  
import pickle  
import pandas as pd
import numpy as np
from PIL import Image
import shap 


# Function to PreProcessing Input Data
def Preprocessing(record, Data):
    """
    Preprocesses a single input record by applying log transformation, imputation (median, mode, KNN),
    label encoding.
    - record: pandas Series (Row of data)
    - Data: The original data (Needed for imputation)
    """
    
    # Log Transformation for specific columns (ensure positive numerical values)
    def log_transform(record, columns):
        for col in columns:
            if isinstance(record[col], (int, float)) and record[col] > 0:  # Ensure value is numeric and positive
                record[col] = np.log(record[col])
            else:
                record[col] = np.nan  # Handle non-positive or non-numeric values gracefully

            
    # List of columns to apply log transformation
    columns_to_transform = ['Blood Glucose Random', 'Blood Urea', 'Serum Creatinine', 'Potassium']
    log_transform(record, columns_to_transform)
   
    
    # Encoding mappings for categorical features
    encodings = {
        "Pus Cell": {"normal": 0, "abnormal": 1},
        "Pus Cell Clumps": {"notpresent": 0, "present": 1},
        "Bacteria": {"notpresent": 0, "present": 1},
        "Hypertension": {"no": 0, "yes": 1},
        "Diabetes Mellitus": {"no": 0, "yes": 1},
        "Coronary Artery Aisease": {"no": 0, "yes": 1},
        "Appetite": {"good": 0, "poor": 1},
        "Peda Edema": {"no": 0, "yes": 1},
        "Aanemia": {"no": 0, "yes": 1},
        "Red Blood Cells": {"normal": 0, "abnormal": 1},
    }

    # Apply encoding
    for col, mapping in encodings.items():
        record[col] = mapping[record[col]]  # Directly map without checking for missing values

    return record



# Function To Apply Independent Discriminational Analysis 
def transform_with_lda(input_data, model_path="trained_ida_model.pkl"):
    """
    Loads a trained LDA model and applies it to transform the input data.

    Parameters:
        input_data (numpy.ndarray or list): Input data to transform (1D list or array).
        model_path (str): Path to the saved LDA model.

    Returns:
        transformed_data (numpy.ndarray): LDA-transformed input.
    """
    # Load the trained LDA model
    with open(model_path, "rb") as file:
        lda = pickle.load(file)

    # Ensure input is a 2D array (required for transform)
    input_data = np.array(input_data).reshape(1, -1)

    # Apply LDA transformation
    transformed_data = lda.transform(input_data)

    return transformed_data



# Loading the Orginal Data
Data = pd.read_csv('PreProcessdData.xls')
Data = Data.drop(['Class' , 'Unnamed: 0'] , axis = 1 ) 

# Loading the Model
with open("Best_model.pkl", "rb") as file:
            ada_model = pickle.load(file)
        
st.set_page_config(layout="wide")  # Make the layout full-width

# Header 
st.markdown("<h1 style= font-family: 'Times New Roman'';'>IntelliKidnye</h1><br><br>", unsafe_allow_html=True)
#About the Project
st.markdown("<h5 style= font-family: 'Times New Roman'';'>About the Project</h5>", unsafe_allow_html=True)

st.markdown("<p style= font-family: 'Times New Roman'';'>This web application is part of a research-driven project focused on Machine Learning & Medical Imaging for Kidney Disease Prediction and Diagnosis. The system leverages advanced deep learning models to assist medical professionals in identifying kidney diseases from CT scan images and structured clinical data.</p>", unsafe_allow_html=True)

# List of Web App Componats
st.markdown("<ul style= font-family: 'Times New Roman': left;'><li>Kidney Disease Prediction: The model analyzes 24 clinical features to predict the likelihood of kidney disease.</li><li>CT Image Classification: A CNN-based model classifies kidney CT scans into four categories: Tumor, Cyst, Stones, or Normal.</li><li>Explainable AI (XAI): Using Grad-CAM, the system highlights critical regions in CT images that influenced the classification, enhancing interpretability.</li><li>Comprehensive Results Dashboard: Users can view predictions, probability scores, and AI-generated heatmaps to understand model decisions.</li></ul>", unsafe_allow_html=True)
st.write("---")  # Separator

# image 
st.sidebar.image('Kid.png')

# Side Bar Menu 
st.sidebar.title('Models')
option = st.sidebar.selectbox("Choose a model" , ["", "Kidney Disease Prediction" , "CT Image Classification" ,"Explainable AI (XAI)" , " Results Dashboard"])

# Selecting Model
if option == "":
    st.markdown("<h5 style= font-family: 'Times New Roman''>Please Select a Model From The Sidebar.", unsafe_allow_html=True)

# If the Option Kidney Disease Prediction
elif option == "Kidney Disease Prediction":
    st.markdown("<h2 style= font-family: 'Times New Roman'>Kidney Disease Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style= font-family: 'Times New Roman''>Clinical Measurements", unsafe_allow_html=True)
    
    # Clinical Measurements
    age = st.number_input('Age', min_value=0, max_value=120, step=1)
    
    col1 , col2 = st.columns(2)
    with col1 : 
        blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, step=1)
        blood_glucose = st.number_input('Blood Glucose Random (mgs/dL)', min_value=0, step=1)
        blood_urea = st.number_input('Blood Urea (mgs/dL)', min_value=0, step=1)
        white_blood_cell_count = st.number_input('White Blood Cell Count (cells/cumm)', min_value=0, step=1)
        red_blood_cell_count = st.number_input('Red Blood Cell Count (millions/cmm)', min_value=0.0, step=0.1)
        
    with col2 :     
        potassium = st.number_input('Potassium (mEq/L)', min_value=0.0, step=0.1)
        haemoglobin = st.number_input('Haemoglobin (gms)', min_value=0.0, step=0.1)
        packed_cell_volume = st.number_input('Packed Cell Volume', min_value=0.0, step=0.1)
        serum_creatinine = st.number_input('Serum Creatinine (mgs/dL)', min_value=0.0, step=0.1)
        sodium = st.number_input('Sodium (mEq/L)', min_value=0.0, step=0.1)
    st.write('---')


    st.markdown("<h3 style= font-family: 'Times New Roman''>Urinalysis/Metabolic Markers", unsafe_allow_html=True)
    
    # Urinalysis/Metabolic Markers
    specific_gravity = st.selectbox('Specific Gravity (The ratio of the density of urine)', ['1.005', '1.010', '1.015', '1.020','1.025']) 

    col3 , col4 = st.columns(2)
    with col3 : 
        albumin = st.selectbox('Albumin (Albumin level in the blood)', ['0', '1',  '2', '3', '4','5'])
    with col4 : 
        sugar = st.selectbox('Sugar ( Sugar level of the patient)', ['0', '1', '2' , '3' ,  '4'  ,'5'])

    st.write('---')

    
    st.markdown("<h3 style= font-family: 'Times New Roman''>Presence of Medical Condition", unsafe_allow_html=True)

    # Presence of Medical Condition
    col5 , col6 = st.columns(2)
    with col5 : 
        hypertension = st.selectbox('Hypertension', ['yes', 'no'])
        diabetes_mellitus = st.selectbox('Diabetes Mellitus', [ 'yes', 'no'])
    with col6 : 
        coronary_artery_disease = st.selectbox('Coronary Artery Disease', ['yes', 'no'])
        aanemia = st.selectbox('Aanemia', ['yes', 'no'])
    
    st.write('---')

    
    st.markdown("<h3 style= font-family: 'Times New Roman''>Symptoms and Clinical Signs", unsafe_allow_html=True)

    # Symptoms and Clinical Signs
    col7 , col8 = st.columns(2)
    with col7 : 
        red_blood_cells = st.selectbox('Red Blood Cells in Urine', ['normal', 'abnormal'])
        pus_cell = st.selectbox('Pus Cells in Urine', ['normal', 'abnormal'])
        appetite = st.selectbox('Appetite', ['good', 'poor'])
    with col8 : 
        pus_cell_clumps = st.selectbox('Pus Cell Clumps in Urine', ['present', 'notpresent'])
        bacteria = st.selectbox('Bacteria in Urine', ['present', 'notpresent'])
        peda_edema = st.selectbox('Peda Edema (Swelling)', ['yes', 'no'])

    st.write("---")



    input_data = pd.Series({
            "Age": age, "Blood Pressure": blood_pressure, "Specific Gravity": specific_gravity, 
            "Albumin": albumin, "Sugar": sugar, "Red Blood Cells": red_blood_cells, 
            "Pus Cell": pus_cell, "Pus Cell Clumps": pus_cell_clumps, "Bacteria": bacteria, 
            "Blood Glucose Random": blood_glucose, "Blood Urea": blood_urea, 
            "Serum Creatinine": serum_creatinine, "Sodium": sodium, "Potassium": potassium, 
            "Haemoglobin": haemoglobin, "Packed Cell Volume": packed_cell_volume, 
            "White Blood Cell Count": white_blood_cell_count, "Red Blood Cell Count": red_blood_cell_count, 
            "Hypertension": hypertension, "Diabetes Mellitus": diabetes_mellitus, 
            "Coronary Artery Aisease": coronary_artery_disease, "Appetite": appetite, 
            "Peda Edema": peda_edema, "Aanemia": aanemia})

   
    
    # Proceed with processing the input data
    processed_input_data = Preprocessing(input_data, Data)
    
    # Apply IDA 
    Ready_data = transform_with_lda(processed_input_data)

    
        
    
    # Prediction Process
    if st.button("Predict"):
        
        required_fields = [age, blood_pressure, blood_glucose, blood_urea, white_blood_cell_count, red_blood_cell_count, 
                           potassium, haemoglobin, packed_cell_volume, serum_creatinine, sodium, specific_gravity, albumin, 
                           sugar, hypertension, diabetes_mellitus, coronary_artery_disease, aanemia]
        
        if any(field == '' or field == 0 for field in required_fields):
            st.error("⚠️ Please fill  all fields!")
            
       # Model Prediction 
       prediction = ada_model.predict(Ready_data)

       if (prediction[0] == 1):
           st.markdown("<h5 style='font-family: Times New Roman;'>The model indicates a likelihood of Chronic Kidney Disease (CKD). Further clinical evaluation is recommended.</h5>", unsafe_allow_html=True)
         
       else: 
           st.markdown("<h5 style='font-family: Times New Roman;'>No significant indicators of Chronic kidney disease (CKD) detected. However, clinical judgment and further assessment may be required.</h5>", unsafe_allow_html=True)
        

        

# If the Option CT Image Classification
elif option == "CT Image Classification":
    st.markdown("<h2 style= font-family: 'Times New Roman''>CT Image Classification</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style= font-family: 'Times New Roman''>Upload a Kidney CT Image</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.success("Image uploaded successfully!")
        

# If the Option Explainable AI (XAI)
elif option == "Explainable AI (XAI)":
    st.markdown("<h2 style= font-family: 'Times New Roman'> Explainable Artificial Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("<p style= font-family: 'Times New Roman'> This section will display the <b> feature importance </b> for the <b> kidney disease prediction model</b> , highlighting which features contribute the most to the model's decision-making. This helps in understanding the impact of different medical parameters, such as blood pressure, serum creatinine, and hemoglobin levels, on the prediction..</p><br>", unsafe_allow_html=True)
    st.markdown("<p style= font-family: 'Times New Roman'>Additionally, this section will include the <b>Grad-CAM heatmap</b> for <b>CT images</b>, providing a visual explanation of which regions in the image were most influential in the model's classification. This enhances interpretability by showing areas of interest for diagnosing kidney conditions such as tumors, cysts, or stones.</p>", unsafe_allow_html=True)

    input_data = st.session_state.input_data  # Retrieve stored input data

    # Apply Prepreocessing
    processed_input_data = Preprocessing(input_data, Data)
    
    # Ensure processed_input_data is a DataFrame
    processed_input_data_df = pd.DataFrame([processed_input_data], columns=Data.columns)

    # Create SHAP Explainer
    explainer = shap.Explainer(ada_model, Data)  # Train SHAP explainer with full dataset
    shap_values = explainer(processed_input_data_df)

    # Waterfall plot (only for one record)
    st.subheader("Feature Importance (Waterfall Plot)")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap_values[0], show=False)
    st.pyplot(fig)

    # Bar plot (Overall Feature Importance)
    st.subheader("Overall Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.bar_plot(shap_values, show=False)
    st.pyplot(fig)

    # Summary plot (if needed)
    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)

    

# If the Option Results Dashboard
elif option == "Results Dashboard":
    st.markdown("<h2 style= font-family: 'Times New Roman';'>Results Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style= font-family: 'Times New Roman';'>Results Dashboard</h2>", unsafe_allow_html=True)


st.write("---")  # Separator

# Function to switch pages
def navigate_to(page):
    st.session_state.page = page


# Create a navigation bar with buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.button("Project Documentation")
with col2:
    st.button("Author's")
with col3:
    st.button("!!!!!!")
with col4:
    st.button("!!!!!")

# Display content based on selected page
