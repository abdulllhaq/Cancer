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
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.
''')
st.write('---')

# Load Data
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    st.error("Error: data.csv not found.  Make sure it's in the same directory.")
    st.stop()

# Preprocessing: Clean column names
df.columns = df.columns.str.strip() #Removes any whitespace around column names, critical to avoid errors

# Prepare Data for Model
X = df.drop('Outcome', axis=1)  #  'Outcome' is the target now!
y = df['Outcome']

# Encode target variable
label_encoder = preprocessing.LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, X_train)  # Train the data in

# User Input Form
def user_input_features():
    age = st.sidebar.slider('Age', int(X['Age'].min()), int(X['Age'].max()), int(X['Age'].mean()))
    radius = st.sidebar.slider('Radius', float(X['Radius'].min()), float(X['Radius'].max()), float(X['Radius'].mean()))
    texture = st.sidebar.slider('Texture', float(X['Texture'].min()), float(X['Texture'].max()), float(X['Texture'].mean()))
    perimeter = st.sidebar.slider('Perimeter', float(X['Perimeter'].min()), float(X['Perimeter'].max()), float(X['Perimeter'].mean()))
    area = st.sidebar.slider('Area', float(X['Area'].min()), float(X['Area'].max()), float(X['Area'].mean()))
    smoothness = st.sidebar.slider('Smoothness', float(X['Smoothness'].min()), float(X['Smoothness'].max()), float(X['Smoothness'].mean()))
    compactness = st.sidebar.slider('Compactness', float(X['Compactness'].min()), float(X['Compactness'].max()), float(X['Compactness'].mean()))
    concavity = st.sidebar.slider('Concavity', float(X['Concavity'].min()), float(X['Concavity'].max()), float(X['Concavity'].mean()))
    concave_points = st.sidebar.slider('Concave points', float(X['Concave points'].min()), float(X['Concave points'].max()), float(X['Concave points'].mean()))
    symmetry = st.sidebar.slider('Symmetry', float(X['Symmetry'].min()), float(X['Symmetry'].max()), float(X['Symmetry'].mean()))
    fractal_dimension = st.sidebar.slider('Fractal Dimension', float(X['Fractal Dimension'].min()), float(X['Fractal Dimension'].max()), float(X['Fractal Dimension'].mean()))

    data = {
        'Age': age,
        'Radius': radius,
        'Texture': texture,
        'Perimeter': perimeter,
        'Area': area,
        'Smoothness': smoothness,
        'Compactness': compactness,
        'Concavity': concavity,
        'Concave points': concave_points,
        'Symmetry': symmetry,
        'Fractal Dimension': fractal_dimension
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
predicted_diagnosis = label_encoder.inverse_transform(prediction)[0] # Decode prediction

st.subheader('Prediction:')
if predicted_diagnosis == 0:
    st.write('Benign (Non-Cancerous)')
else:
    st.write('Malignant (Cancerous)')

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
st.write('Disclaimer: This is for educational purposes only. Consult a doctor for medical advice.')

#Remove this since the link is not working
#st.sidebar.subheader("An article about this app: https://proskillocity.blogspot.com/2021/06/breast-cancer-detection-web-app.html")
