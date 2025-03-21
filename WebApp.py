import streamlit as st  
import pickle  
import pandas as pd
import numpy as np
from PIL import Image
import shap 
import psycopg2


# Function to PreProcessing Input Data
def Preprocessing(record, Data):

    def log_transform(record, columns):
        for col in columns:
            if isinstance(record[col], (int, float)) and record[col] > 0:  # Ensure value is numeric and positive
                record[col] = np.log(record[col])
            else:
                record[col] = np.nan  
                
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
        record[col] = mapping[record[col]]  
        
    return record



# Function To Apply Independent Discriminational Analysis 
def transform_with_lda(input_data, model_path="trained_ida_model.pkl"):
    
    # Load the trained LDA model
    with open(model_path, "rb") as file:
        lda = pickle.load(file)

    # Ensure input is a 2D array (required for transform)
    input_data = np.array(input_data).reshape(1, -1)

    # Apply LDA transformation
    transformed_data = lda.transform(input_data)

    return transformed_data

# Function to create a connection to the SQLite database
def create_connection(DatabaseURL):
    try:
        conn = psycopg2.connect(DatabaseURL)
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# Function to insert Data into Database 
def insert_data(conn, data_tuple):
    cursor = conn.cursor()
    insert_query = '''
    INSERT INTO ClinicalMeasurements (
        Age, BloodPressure, BloodGlucoseRandom, BloodUrea, WhiteBloodCellCount,
        RedBloodCellCount, Potassium, Haemoglobin, PackedCellVolume, SerumCreatinine,
        Sodium, SpecificGravity, Albumin, Sugar, Hypertension, DiabetesMellitus,
        CoronaryArteryDisease, Anemia, RedBloodCellsInUrine, PusCellsInUrine,
        Appetite, PusCellClumpsInUrine, BacteriaInUrine, PedalEdema, Class
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    '''
    # Execute query
    cursor.execute(insert_query, data_tuple)
    conn.commit()
    

# Database URL 
DatabaseURL = "postgresql://neondb_owner:npg_MCBW0Q8pqvVJ@ep-tight-rain-a55tsq6b-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"

# Loading the Orginal Data 
Data = pd.read_csv('PreProcessdData.xls')
Data = Data.drop(['Class' , 'Unnamed: 0'] , axis = 1 ) 

# Loading the Model
with open("Best_model.pkl", "rb") as file:
            ada_model = pickle.load(file)

# Loading Model WithOut IDA (XAI)
with open("Adaboost_shap_explainer.pkl", "rb") as file:
            ada_model_XAI = pickle.load(file)

# Make the layout full-width
st.set_page_config(layout="wide")  

# Header 
st.markdown("<h1 style= font-family: 'Times New Roman'';'>IntelliKidnye</h1><br><br>", unsafe_allow_html=True)
# About the Project
st.markdown("<h5 style= font-family: 'Times New Roman'';'>About the Project</h5>", unsafe_allow_html=True)

st.markdown("<p style= font-family: 'Times New Roman'';'>This web application is part of a research-driven project focused on Machine Learning & Medical Imaging for Kidney Disease Prediction and Diagnosis. The system leverages advanced deep learning models to assist medical professionals in identifying kidney diseases from CT scan images and structured clinical data.</p>", unsafe_allow_html=True)
st.markdown(
    """
            <p style= font-family: 'Times New Roman'';'>
            <b>Kidney Disease Prediction</b> 
            The model analyzes 24 clinical features to predict the likelihood of Chronic Kidney Disease (CKD). 
            Using <b>Explainable AI (XAI)</b>, the system provides <b>feature importance</b> scores, highlighting the top clinical factors (e.g., Haemoglobin, Specific Gravity, Sodium) that influenced the prediction. 
            This helps doctors understand the model's reasoning and make informed decisions.
            </p>

            <p style= font-family: 'Times New Roman'';'>
            <b>CT Image Classification</b> 
            A CNN-based model classifies kidney CT scans into four categories: <b>Tumor</b>, <b>Cyst</b>, <b>Stones</b>, or <b>Normal</b>. 
            Using <b>Grad-CAM (Gradient-weighted Class Activation Mapping)</b>, the system highlights critical regions in the CT images that influenced the classification. 
            This enhances interpretability and helps radiologists identify areas of concern.
            </p>

          
    """, 
    unsafe_allow_html=True
)
# Seperator
st.write('---')

# image 
st.sidebar.image('Kid.png')

# Button Style
st.markdown(
    """
    <style>
    .stButton>button {
        width: 200px;  
        height: 50px;  
        font-size: 18px;  
        margin: 0 auto;  
        display: block;  
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Side Bar Menu 
option = st.sidebar.selectbox('' , ["Choose a model", "Kidney Disease Prediction" , "CT Image Classification"  , " Results Dashboard"])

# Selecting Model
if option == "Choose a model":
    st.markdown("<center><h5 style= font-family: 'Times New Roman'>Please select a model from the sidebar to get started ", unsafe_allow_html=True)

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

    # Urinalysis/Metabolic Markers
    st.markdown("<h3 style= font-family: 'Times New Roman''>Urinalysis/Metabolic Markers", unsafe_allow_html=True)
    
    specific_gravity = st.selectbox('Specific Gravity (The ratio of the density of urine)', ['1.000','1.005', '1.010', '1.015', '1.020','1.025', '1.030']) 

    col3 , col4 = st.columns(2)
    with col3 : 
        albumin = st.selectbox('Albumin (Albumin level in the blood)', ['0', '1',  '2', '3', '4','5'])
    with col4 : 
        sugar = st.selectbox('Sugar ( Sugar level of the patient)', ['0', '1', '2' , '3' ,  '4'  ,'5'])

    st.write('---')

    # Presence of Medical Condition
    st.markdown("<h3 style= font-family: 'Times New Roman''>Presence of Medical Condition", unsafe_allow_html=True)

    col5 , col6 = st.columns(2)
    with col5 : 
        hypertension = st.selectbox('Hypertension', ['yes', 'no'])
        diabetes_mellitus = st.selectbox('Diabetes Mellitus', [ 'yes', 'no'])
    with col6 : 
        coronary_artery_disease = st.selectbox('Coronary Artery Disease', ['yes', 'no'])
        aanemia = st.selectbox('Aanemia', ['yes', 'no'])
    
    st.write('---')

    # Symptoms and Clinical Signs
    st.markdown("<h3 style= font-family: 'Times New Roman''>Symptoms and Clinical Signs", unsafe_allow_html=True)

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



    if st.button("Predict"):
        required_fields = [age, blood_pressure, blood_glucose, blood_urea, white_blood_cell_count, red_blood_cell_count, 
                               potassium, haemoglobin, packed_cell_volume, serum_creatinine, sodium, specific_gravity, albumin, 
                               sugar, hypertension, diabetes_mellitus, coronary_artery_disease, aanemia]
            #????????
        if any(field == 0 for field in required_fields):
                st.error("⚠️ Please fill  all fields!")
        else:   
            # Input Data
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

            # Prediction
            prediction = ada_model.predict(Ready_data)
            
            # Example new record ( For the XAI Model)
            new_record = np.array([list(processed_input_data)]) 
            new_record = new_record.astype(np.float64)

            # Convert to pandas DataFrame ( For the XAI Model)
            new_record_df = pd.DataFrame(new_record, columns=Data.columns)
            
            # Generate SHAP values for explanation
            shap_values = ada_model_XAI(new_record)

            top_features = np.argsort(-np.abs(shap_values.values[0]))[:3]
            explanation_markdown = ''
            for feature in top_features:
                explanation_markdown += "\n".join([f"- **{new_record_df.columns[feature]}** (Impact: {shap_values.values[0][feature]:.2f})\n"])                         
                    
            # Display the prediction result
            if prediction[0] == 1:
                st.markdown("<p>The model has identified a likelihood of Chronic Kidney Disease (CKD) based on the patient's data.\nBelow is a breakdown of the top 3 features contributing to this diagnosis, along with their relative impact and clinical significance. </h5>", unsafe_allow_html=True)
                st.markdown(explanation_markdown)
                st.markdown("<p> For more detailed analysis or to investigate possible structural causes or forms of CKD (e.g., kidney stones, cysts, or abnormalities), please proceed to the <b>CT Image Analysis model</b>.</h5>", unsafe_allow_html=True)
                                
            else:
                st.markdown("<p>The model has assessed the patient's data and found no significant likelihood of Chronic Kidney Disease (CKD).\nBelow is a breakdown of the top 3 features contributing to this diagnosis, along with their relative impact and clinical significance.</h5>", unsafe_allow_html=True)
                st.markdown(explanation_markdown)
                st.markdown("<p>For further analysis or to rule out any potential early signs of kidney-related issues (e.g., kidney stones, cysts, or other abnormalities), please proceed to the <b>CT Image Analysis model</b>.</p>", unsafe_allow_html=True)

                            
            # Add the Class to the Record (data_tuple)
            Class = str(prediction[0])
            data_tuple = [age, blood_pressure, blood_glucose, blood_urea, white_blood_cell_count,
              red_blood_cell_count, potassium, haemoglobin, packed_cell_volume, serum_creatinine,
              sodium, specific_gravity, albumin, sugar, hypertension, diabetes_mellitus,
              coronary_artery_disease, aanemia, red_blood_cells, pus_cell,
              appetite, pus_cell_clumps, bacteria, peda_edema, Class]
            data_tuple = tuple(data_tuple)

            # Connect with the Database and inser tuple
            conn = create_connection(DatabaseURL)
            insert_data(conn, data_tuple)
         

    
# If the Option CT Image Classification
elif option == "CT Image Classification":
    st.markdown("<h2 style= font-family: 'Times New Roman''>CT Image Classification</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style= font-family: 'Times New Roman''>Upload a Kidney CT Image</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.success("Image uploaded successfully!")
        

# If the Option Explainable AI (XAI)
# elif option == "Explainable AI (XAI)":
#     st.markdown("<h2 style= font-family: 'Times New Roman'> Explainable Artificial Intelligence</h2>", unsafe_allow_html=True)
#     st.markdown("<p style= font-family: 'Times New Roman'> This section will display the <b> feature importance </b> for the <b> kidney disease prediction model</b> , highlighting which features contribute the most to the model's decision-making. This helps in understanding the impact of different medical parameters, such as blood pressure, serum creatinine, and hemoglobin levels, on the prediction.</p><br>", unsafe_allow_html=True)
    
#     st.markdown("<p style= font-family: 'Times New Roman'>Each health factor like (hemoglobin level, serum creatinine, diabetes, etc.) is represented by the associated bar. The length of these bars denotes the extent to which each health factor has been incorporated into the kidney disease prediction. Thus, the longer the bar, the more impact that factor has been upon the prediction.This is combined with Haemoglobin (+0.22), which is mainly the most effective one. If your hemoglobin levels are low or high, it has a direct effect on how the model thinks. Serum creatinine (+0.08): This is a marker of kidney function. But it goes a long way in the health outcome prediction, Diabetes Mellitus (+0.06): If you are a diabetic, it greatly affects your prediction, Specific Gravity (+0.05): This one is linked to urine concentration, which may reflect kidney functional problems. The addition of other 15 features results in a joint SHAP value of +0.06, which tells that although these other features are contributing to prediction, their respective contributions are rather small, even having the consideration of these features being in the model in some situations; their overall significance is minor. So, Hemoglobin and serum creatinine values are the most crucial and in response, your doctor might touch base on those values while diagnosing or under treatment.</p><br>", unsafe_allow_html=True)

#     # image 
#     st.image('SHAP Summary Bar Chart.png')

#     st.write('---')
#     st.markdown("<p style= font-family: 'Times New Roman'>Predicted outcome by AI model reason for prediction for you, Baseline Prediction (E[f(X)]=0.59): Mean or baseline prediction for all patients before assessing individual health variables=Red Bar (+ values) Higher=pushes prediction higher. Increases chance that this condition is predicted. Blue Bars (- values) lower-pushed prediction. Decreases chance that this condition is predicted.Diabetes Mellitus (+0.24):This increases the prediction significantly resorted diabetes means it is an important factor for your condition.In case of Hypertension (+0.22):This means that blood pressure raises strong odds through prediction, Meaning that prediction seems to have close association between this condition-hypertension and condition prediction. Blood Urea (+0.17):A high concentration of urea in the blood suggests kidney damage, Building in prediction towards confidence. Potassium (+0.14):Having low or high abnormal potassium levels affects your kidneys. In addition, this makes prediction rise in the model. Haemoglobin (-0.17):haemoglobin levels tend to adversely affect prediction, This can be interpreted as your hemoglobin might be normal or higher, thus lowering the ratio of having the condition. Serum Creatinine (-0.1):Creatinine is one major key kidney functional parameters, since it is causing the reduction of prediction. Other minor factors such as Specific Gravity (-0.06) and Blood Glucose Random (-0.04) have some lesser influences, but continue to positively contribute toward predicting. Most important factors for health risk are diabetes and hypertension, however blood urea and potassium levels also contribute to increasing the prediction. Haemoglobin and serum creatinine levels are reducing the prediction, which shows it may be normal for you. This information might be shared with your doctor to manage diabetes, blood pressure.</p><br>", unsafe_allow_html=True)
#     st.image('XAI ( Feature Importance).png')


#     st.markdown("<p style= font-family: 'Times New Roman'>Additionally, this section will include the <b>Grad-CAM heatmap</b> for <b>CT images</b>, providing a visual explanation of which regions in the image were most influential in the model's classification. This enhances interpretability by showing areas of interest for diagnosing kidney conditions such as tumors, cysts, or stones.</p>", unsafe_allow_html=True)

    
   
# If the Option Results Dashboard

elif option == "Results Dashboard":
    st.markdown("<h2 style= font-family: 'Times New Roman';'>Results Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style= font-family: 'Times New Roman';'>Results Dashboard</h2>", unsafe_allow_html=True)


st.write("---")  # Separator

# Create a navigation bar with buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.button("Project Documentation")
with col2:
    st.button("Medical Report")
with col3:
    st.button("Performance Evaluation")
with col4:
    Author_button = st.button("Author's")

st.write('---')

st.write('\n\n')
st.write('\n\n')

# Display the contact information when the "Contact" button is clicked
if Author_button:
    st.markdown(
        '''
        <p style="font-family: 'Times New Roman', Times, serif; font-size: 16px; text-align: center;">
            Mones Nazih Ksasbeh - <a href="mailto:moksasbeh@gmail.com">Email</a>  &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp;|&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp;
            Yazan Amjed Mansour - <a href="mailto:am5294690@gmail.com">Email</a> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp;| &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp;
            Basel Mwafq Hammo - <a href="mailto:basel.11hammo@gmail.com">Email</a> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        </p>
        ''',
        unsafe_allow_html=True
    )

