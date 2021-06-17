#import statements
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from PIL import Image



#App description
st.markdown('''
# Cancer Detector 
This app detects if you have diabetes based on Machine Learning!
- App built by Pranav Sawant and Anshuman Shukla of Team Skillocity.
- Dataset resource: Pima Indian Datset (United States National Institutes of Health). 
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.  
''')
st.write('---')

df = pd.read_csv(r'data.csv')

#titles
st.sidebar.header('Patient Data')
st.subheader('Training Dataset')
st.write(df.describe())


#train data. Fun!
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


#User reports
def user_report():
  Age = st.sidebar.slider('Age', 0,100, 54)
  Radius = st.sidebar.slider('Radius', 0,250, 120 )
  Texture = st.sidebar.slider('Texture', 0,850, 90 )
  Perimeter = st.sidebar.slider('Perimeter', 0,300, 85 )
  Area = st.sidebar.slider('Area', 0,70, 22 )
  Smoothness = st.sidebar.slider('Smoothness', 0.0,3.0, 0.8 )
  Compactness = st.sidebar.slider('Compactness', 21,120, 55 )
  Concavity = st.sidebar.slider('Concavity', 0,10, 1 )
  Concave_points = st.sidebar.slider('Concave points', 0,100, 35 )
  Symmetry = st.sidebar.slider('Symmetry', 0,100, 35 )
  Fractal_Dimension = st.sidebar.slider('Fractal Dimension', 0,100, 35 )
  
  
  
  
  
  user_report_data = {
      'Age':glucose,
      'insulin':insulin,
      'bp':bp,
      'bmi':bmi,
      'dpf':dpf,
      'age':age,
      'pregnancies':pregnancies,
      'skinthickness':skinthickness,
         
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data





user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)





rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)



#Visualizations, this is where the beauty begins.
st.title('Graphical Patient Report')



if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

#Good old glucose
st.header('Glucose Value Graph (Yours vs Others)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='Purples')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,250,20))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)


#Insulin
st.header('Insulin Value Graph (Yours vs Others)')
fig_insulin = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rainbow')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_insulin)


#Famous saying BP
st.header('Blood Pressure Value Graph (Yours vs Others)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Blues')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,320,20))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


#Did'nt even know this before nutrition training 
st.header('BMI Value Graph (Yours vs Others)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='Greens')
ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,75,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)


#Something new, cool
st.header('DPF Value Graph (Yours vs Others)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='rocket')
ax14 = sns.scatterplot(x = user_data['age'], y = user_data['dpf'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,3.2,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)


#Don't even know how thats related to diabetes.The dataset was females only though
st.header('Pregnancy count Graph (Yours vs Others)')
fig_pregn = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'magma')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_pregn)


#Wonder how people measure that 
st.header('Skin Thickness Value Graph (Yours vs Others)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Reds')
ax8 = sns.scatterplot(x = user_data['age'], y = user_data['skinthickness'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


#Finally!
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'Congratulations, you are not Diabetic'
else:
  output = 'Unfortunately, you are Diabetic'
st.title(output)

#Most important for users
st.subheader('Lets raise awareness for diabetes and show our support for diabetes awareness and help many patients around the world.')
st.write("World Diabetes Day: 14 November")

st.sidebar.subheader("""An article about this app: https://proskillocity.blogspot.com/2021/04/official-launch-of-our-first-web-app.html""")
st.write("Dataset citation : Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988).  Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.")
st.write("Original owners of the dataset: Original owners: National Institute of Diabetes and Digestive and Kidney Diseases (b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu) Research Center, RMI Group Leader Applied Physics Laboratory The Johns Hopkins University Johns Hopkins Road Laurel, MD 20707 (301) 953-6231 © Date received: 9 May 1990")
st.write("Dataset License: Open Data Commons Public Domain Dedication and License (PDDL)")

st.write("Disclaimer: This is just a learning project based on one particular dataset so please do not depend on it to actually know if you have diabetes or not. It might still be a false positive or false negative. A doctor is still the best fit for the determination of such diseases.")
image = Image.open('killocity (3).png')

st.image(image, use_column_width=True)
