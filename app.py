import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Import numpy
from PIL import Image  # Import the Image module

# App Title
st.title('Breast Cancer Prediction App')

# About Section
st.markdown('''
# Breast Cancer Detector
This app detects if you have Breast Cancer based on Machine Learning!
- App built by Abdul Haq of Team Skillocity.
- Dataset Creators: 
    - Dr. William H. Wolberg, General Surgery Dept. University of Wisconsin, Clinical Sciences Center Madison, WI 53792
    - W. Nick Street, Computer Sciences Dept. University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
    - Olvi L. Mangasarian, Computer Sciences Dept. University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.  
''')
st.write('---')

# Load Data
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    st.error("Error: data.csv not found.  Make sure it's in the same directory.")
    st.stop()

# Rename 'Unnamed: 32' column and drop it if it exists
if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)

# Preprocessing
df = df.drop(['id'], axis=1, errors='ignore') # drop the 'id' column since it is irrelevant

# Data Summary
st.sidebar.header('Patient Data Input')
st.subheader('Dataset Overview')
st.write(df.describe())

# Prepare Data for Model
X = df.drop('diagnosis', axis=1)  # Use 'diagnosis' as the target
y = df['diagnosis']

# Encode target variable
label_encoder = preprocessing.LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# User Input Form
def user_input_features():
    age = st.sidebar.slider('Age', int(X['radius_mean'].min()), int(X['radius_mean'].max()), int(X['radius_mean'].mean()))
    radius_mean = st.sidebar.slider('Radius (Mean)', float(X['radius_mean'].min()), float(X['radius_mean'].max()), float(X['radius_mean'].mean()))
    texture_mean = st.sidebar.slider('Texture (Mean)', float(X['texture_mean'].min()), float(X['texture_mean'].max()), float(X['texture_mean'].mean()))
    perimeter_mean = st.sidebar.slider('Perimeter (Mean)', float(X['perimeter_mean'].min()), float(X['perimeter_mean'].max()), float(X['perimeter_mean'].mean()))
    area_mean = st.sidebar.slider('Area (Mean)', float(X['area_mean'].min()), float(X['area_mean'].max()), float(X['area_mean'].mean()))
    smoothness_mean = st.sidebar.slider('Smoothness (Mean)', float(X['smoothness_mean'].min()), float(X['smoothness_mean'].max()), float(X['smoothness_mean'].mean()))
    compactness_mean = st.sidebar.slider('Compactness (Mean)', float(X['compactness_mean'].min()), float(X['compactness_mean'].max()), float(X['compactness_mean'].mean()))
    concavity_mean = st.sidebar.slider('Concavity (Mean)', float(X['concavity_mean'].min()), float(X['concavity_mean'].max()), float(X['concavity_mean'].mean()))
    concave_points_mean = st.sidebar.slider('Concave Points (Mean)', float(X['concave points_mean'].min()), float(X['concave points_mean'].max()), float(X['concave points_mean'].mean()))
    symmetry_mean = st.sidebar.slider('Symmetry (Mean)', float(X['symmetry_mean'].min()), float(X['symmetry_mean'].max()), float(X['symmetry_mean'].mean()))
    fractal_dimension_mean = st.sidebar.slider('Fractal Dimension (Mean)', float(X['fractal_dimension_mean'].min()), float(X['fractal_dimension_mean'].max()), float(X['fractal_dimension_mean'].mean()))

    data = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
predicted_diagnosis = label_encoder.inverse_transform(prediction)[0] # Decode prediction

st.subheader('Prediction:')
if predicted_diagnosis == 'M':
    st.write('Malignant (Cancerous)')
else:
    st.write('Benign (Non-Cancerous)')

# Model Performance
st.subheader('Model Accuracy:')
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f'{accuracy * 100:.2f}%')

# Visualization (Example: Feature Importance)
st.subheader('Feature Importance:')
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
st.pyplot(plt) # display plot in streamlit

# Footer
st.write('App built by Abdul Haq.')
st.write('Dataset citation : W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993. O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and prognosis via linear programming. Operations Research, 43(4), pages 570-577, July-August 1995.")
st.write("Dataset License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)")
st.write('Disclaimer: This is for educational purposes only. Consult a doctor for medical advice.')

# Remove the image
#image = Image.open('killocity (3).png')
#st.image(image, use_column_width=True)

#Remove this since the link is not working
#st.sidebar.subheader("An article about this app: https://proskillocity.blogspot.com/2021/06/breast-cancer-detection-web-app.html")
