import streamlit as st  
import pickle  
import pandas as pd
import numpy as np
from PIL import Image 
import io 
import shap 
import psycopg2
from pymongo import MongoClient
import gridfs   
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from lime import lime_image
from skimage.segmentation import mark_boundaries

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
    

# Preprocess image function
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.convert("RGB")  # Ensure the image is in RGB mode (3 channels)
    img = img.resize((224, 224))  # Resize image to match EfficientNetV2B0 input shape
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)
    img_array = preprocess_input(img_array)  # Apply EfficientNetV2B0 preprocessing
    return  img_array  # Return both the image and its numpy array


def generate_lime_explanation(image_array):
    explainer = lime_image.LimeImageExplainer()

    # LIME expects float64
    explanation = explainer.explain_instance(
        image_array.astype('double'),
        CT_Model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    # Get the mask and overlay it on the image
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    # Visualize with boundaries
    img_boundaries = mark_boundaries(temp / 255.0, mask)

    return img_boundaries


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
st.markdown("<h1 style= font-family: 'Times New Roman';>IntelliKidnye</h1><br><br>", unsafe_allow_html=True)
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
            Using <b>LIME (Local Interpretable Model-agnostic Explanations) </b>, the system highlights critical regions in the CT images that influenced the classification. 
            This enhances interpretability and helps radiologists identify areas of concern.
            </p>

          
    """, 
    unsafe_allow_html=True
)
# Seperator
st.write('---')
st.sidebar.write('')

st.markdown("""
<style>
    /* Style all buttons inside sidebar */
    .stButton > button, .stDownloadButton > button {
        background-color: #ffffff;
        font-size: 16px;
        border-radius: 7px;
        border: 1px solid #ccc;
        width: 246px;
        height: 40px;
        margin: 0 auto;
        display: block;
    }

    /* Style the sidebar background */
    .css-1d391kg {
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.image("Kid.png" , width = 250)
option = st.sidebar.selectbox(
    '', 
    ["Choose a Model", "Kidney Disease Prediction", "CT Image Classification"]
)
st.sidebar.write('---')

# Create a sidebar with 1 column and 4 rows for buttons
with st.sidebar: 
    with open("IntelliKidney.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(
        label="Project Documentation",
        data=PDFbyte,
        file_name="IntelliKidney.pdf",
        mime="application/pdf",
        key="btn_1"
    )

    System_Performance = st.button("System Performance", key="btn_3")
    Author = st.button("Author's", key="btn_4")



# Selecting Model
if option == "Choose a Model":
    st.markdown("<center><h3 style= font-family: 'Times New Roman'>Please select a model from the sidebar to get started ", unsafe_allow_html=True)
    
elif System_Performance:
        st.markdown(
            """
            <div style= font-family: 'Times New Roman'';; font-size: 18px; text-align: justify;">
            <h3>System Performance Report</h3>
                <p>
                The diagnostic system developed in this project includes two advanced artificial intelligence models designed to assist in the early detection and classification of kidney-related conditions. The first model is a structured-data-based predictive system trained using the AdaBoost classifier. It analyzes patient clinical information, such as laboratory test results, symptoms, and medical history, to predict the likelihood of kidney disease. This model demonstrated exceptional performance, achieving a training accuracy of 95.6% and a testing accuracy of 95.2%. These figures indicate that the model maintains a strong ability to generalize from training data to unseen patient cases. Furthermore, its precision reached 93.8%, meaning that the majority of patients identified as having kidney disease were correctly diagnosed. Most notably, the model achieved a recall (sensitivity) of 96.9%, which is critically important in medical applications. A high recall means that the system effectively detects nearly all actual kidney disease cases, minimizing the risk of overlooking affected patients. The F1 score, which reflects a balance between precision and recall, was 95.3%, confirming the overall reliability of this model in clinical scenarios.
                </p>
            """,
            unsafe_allow_html=True  # <-- THIS is important
        )
        st.markdown(
            """
            <div style= font-family: 'Times New Roman'';; font-size: 18px; text-align: justify;">
                <p>
                The second model focuses on image-based diagnosis using kidney CT scans and leverages deep learning through the EfficientNet architecture. This model was fine-tuned on thousands of labeled CT images categorized into four classes: Cyst, Normal, Stone, and Tumor. Upon evaluation, it achieved a test accuracy of 95.9%, indicating a high level of performance in image classification tasks. The model demonstrated excellent sensitivity across all four diagnostic categories: 98% for Cyst detection, 94% for Normal images, 97% for Stone, and 96% for Tumor identification. These results highlight the model’s robustness and precision in differentiating between subtle anatomical features on CT scans. For example, it can reliably distinguish between benign cysts and potentially dangerous tumors, which is essential for guiding timely and accurate medical intervention. Overall, both models show a high degree of accuracy and reliability, offering a promising decision-support tool for physicians and radiologists in the early identification and classification of kidney conditions.
                </p>
            </div>
            """,
            unsafe_allow_html=True  # <-- THIS is important
        )
        st.write('---')    
# If the Option Kidney Disease Prediction
elif option == "Kidney Disease Prediction":
    st.markdown("<center><h3 style= font-family: 'Times New Roman'>Kidney Disease Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style= font-family: 'Times New Roman''>Clinical Measurements", unsafe_allow_html=True)

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
             
            # Database URL 
            DatabaseURL = "postgresql://neondb_owner:npg_MCBW0Q8pqvVJ@ep-tight-rain-a55tsq6b-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
                # Connect with the Database and inser tuple
            conn = create_connection(DatabaseURL)
            insert_data(conn, data_tuple)
         
    
# Add the Prediction button
elif option == "CT Image Classification":
    
    st.markdown("<center><h3 style='font-family: Times New Roman'; text-align: center;>CT Image Classification</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='font-family: Times New Roman'; text-align: center;>Upload a Kidney CT Image</h5>", unsafe_allow_html=True)

    # Loading the Transfer learning model
    CT_Model = tf.keras.models.load_model('fine_tuned_EfficientNetV2B0_model.h5') 
    # File uploader and image classification
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    prediction_button = st.button("Predict")  # Button for prediction
    # Display a button for prediction
    if uploaded_file is not None:
        st.markdown("<br><br>", unsafe_allow_html=True)
        prediction_button = st.button("Predict")  # Button for prediction
        if prediction_button:
            # Preprocess the image
            img_array = preprocess_image(uploaded_file)
        
            # Make the prediction
            predictions = CT_Model.predict(img_array)  # Ensure shape (1, 224, 224, 3)
            class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
            predicted_class = class_names[np.argmax(predictions)]  # Get the predicted class label
            
            # Open image and get format
            image = Image.open(uploaded_file)
            image_format = image.format  # Example: 'PNG', 'JPEG', etc.
            
            # Convert the image to bytes using its original format
            image_bytes_io = io.BytesIO()
            image.save(image_bytes_io, format=image_format)
            image_bytes_io.seek(0)  # Rewind to the beginning

            # MongoDB connection
            Client = MongoClient("mongodb+srv://Mones:Ksasbeh@cluster0.cmk64.mongodb.net/CT_Images?retryWrites=true&w=majority")
            MongoDB = Client['CT_Images']
            fs_cyst = gridfs.GridFS(MongoDB, collection="Cyst")
            fs_normal = gridfs.GridFS(MongoDB, collection="Normal")
            fs_stone = gridfs.GridFS(MongoDB, collection="Stone")
            fs_tumor = gridfs.GridFS(MongoDB, collection="Tumor")
            
            # Display prediction and store image based on prediction
            if predicted_class == 'Normal':
                st.markdown("<h4 style='font-family: Times New Roman;'>Normal Kidney</h3>", unsafe_allow_html=True)
                st.markdown("<p>The kidney appears healthy with no visible signs of abnormalities. There are no cysts, stones, or masses detected, indicating normal renal function.</p>", unsafe_allow_html=True)
                file_id = fs_normal.put(image_bytes_io, filename='normal_image.jpg')
                
            elif predicted_class == 'Cyst':
                st.markdown("<h4 style='font-family: Times New Roman;'>Kidney Cyst</h3>", unsafe_allow_html=True)
                st.markdown("<p>A cyst is detected in the kidney. Simple renal cysts are typically benign and often don't require treatment, but their size and any associated symptoms may require follow-up imaging.</p>", unsafe_allow_html=True)
                file_id = fs_cyst.put(image_bytes_io, filename='cyst_image.jpg')

            elif predicted_class == 'Stone':
                st.markdown("<h4 style='font-family: Times New Roman;'>Kidney Stone</h3>", unsafe_allow_html=True)
                st.markdown("<p>Kidney stones are present, which may cause pain or discomfort. The stones' size, location, and potential for obstruction should be evaluated to determine appropriate management options.</p>", unsafe_allow_html=True)
                file_id = fs_stone.put(image_bytes_io, filename='stone_image.jpg')
                
            elif predicted_class == 'Tumor':
                st.markdown("<h4 style='font-family: Times New Roman;'>Kidney Tumor</h3>", unsafe_allow_html=True)
                st.markdown("<p>A mass suggesting a renal tumor is detected. Further imaging and possibly biopsy are needed to assess the tumor's nature, whether benign or malignant, and plan further action.</p>", unsafe_allow_html=True)
                file_id = fs_tumor.put(image_bytes_io, filename='tumor_image.jpg') 

            
            image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224))  # Adjust target_size
            image = np.array(image)
            
            st.write("Generating  explanation... please wait ⏳")
            lime_img = generate_lime_explanation(image)
            # Create three columns with one centered
            col1, col2, col3 = st.columns([1, 2, 1])  # This will create a layout with columns
            with col2:
                st.image(lime_img, caption="Highlighted areas that influenced the model's decision", width=400)
                
    else : 
        st.error("⚠️ Please Uploaded The CT Image!")

    
st.write("---")  # Separator  
st.write('\n\n')
st.write('\n\n')

# Display the contact information when the "Contact" button is clicked
if Author:
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
    

