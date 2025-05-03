import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

class Prediction:
  def __init__(self,df):
    self.df = df

  def Grouping_degree(self,degree,df):
    for i in degree:
      if i.startswith('B') or i=='MBBS' or i=='LLB':
        df['Degree']=df['Degree'].replace(i,'Bachelors')
      elif i.startswith('M') or i=='LLM':
        df['Degree']=df['Degree'].replace(i,'Masters')
      elif i.startswith('P'):
        df['Degree']=df['Degree'].replace(i,'Doctorate')
      else:
        pass
    return df

  def clean_data(self,df):
    # Drop ID and Name columns in the dataframe
    df.drop(['id','Name'],axis = 1, inplace = True)

    # Handling Null values in numerical data columns
    df['Academic Pressure']=df['Academic Pressure'].fillna(0.0)
    df['Work Pressure']=df['Work Pressure'].fillna(0.0)
    df['CGPA']=df['CGPA'].fillna(0.0)
    df['Study Satisfaction']=df['Study Satisfaction'].fillna(0.0)
    df['Job Satisfaction']=df['Job Satisfaction'].fillna(0.0)
    df['Financial Stress']=df['Financial Stress'].fillna(0.0)

    # Clean Dietary Habits
    df['Dietary Habits']= df['Dietary Habits'].apply(lambda x: 'Unhealthy' if x in ['Less than Healthy','No Healthy',
                                                                                    'Less Healthy'] else x)
    df['Dietary Habits']= df['Dietary Habits'].apply(lambda x: 'Healthy' if x == 'More Healthy' else x)
    df['Dietary Habits']= df['Dietary Habits'].apply(lambda x: x if x in ['Healthy','Unhealthy','Moderate']
                                                    else 'Other')

    # Clean City Data - Replace the misspelled City name with correct value and
    # change the names to Other for Cities with less occurances
    city=list(df['City'].unique())
    df['City']=df['City'].replace(['Galesabad','Khaziabad'],'Ghaziabad')
    df['City']=df['City'].replace(['Itheg','Ithal'],'Ithalar')
    df['City']=df['City'].replace(['Tolkata','Molkata'],'Kolkata')
    df['City']=df['City'].replace(['Nalyan','Less than 5 Kalyan'],'Kalyan')
    df['City']=df['City'].replace(['Less Delhi'],'Delhi')

    freq = df['City'].value_counts(normalize=True)
    df['City'] = df['City'].apply(lambda x: x if freq[x] > 0.0001 else 'Other')

    # Sleep duration
    freq = df['Sleep Duration'].value_counts(normalize=True)
    df['Sleep Duration'] = df['Sleep Duration'].apply(lambda x: x if freq[x] > 0.003 else 'Other')

    # Degree
    df['Degree']=df['Degree'].fillna('Other')
    freq = df['Degree'].value_counts(normalize=True)
    df['Degree'] = df['Degree'].apply(lambda x: x if freq[x] > 0.003 else 'Other')
    degree = list(df['Degree'].unique())
    df=self.Grouping_degree(degree,df)

    # Profession
    df["Profession"] = df.apply(lambda x: "Student" if x["Working Professional or Student"] == "Student"
                                and str(x["Profession"]) == "nan"
                                else x["Profession"], axis = 1)
    df['Profession']=df['Profession'].fillna('Other')
    df['Profession']=df['Profession'].replace('City Manager','Manager')
    df['Profession']=df['Profession'].replace('Dev','Software Engineer')
    df['Profession']=df['Profession'].replace('Medical Doctor','Doctor')
    df['Profession']=df['Profession'].replace('Finanancial Analyst','Financial Analyst')
    freq = df['Profession'].value_counts(normalize=True)
    df['Profession'] = df['Profession'].apply(lambda x: x if freq[x] > 0.003 else 'Other')

    return df

  def Preprocess(self):
    tr_data = pd.read_csv("/content/drive/MyDrive/Project-5-Predict-depression/mental_health/train.csv")
    tr_data.dropna(subset=['Degree','Financial Stress','Dietary Habits'],inplace=True)
    tr_data.drop(tr_data[(tr_data['Profession'].isnull()) & (tr_data['Working Professional or Student']!='Student')].index,inplace=True)
    tr_data.drop(tr_data[(tr_data['Academic Pressure'].isnull()) & (tr_data['Working Professional or Student']=='Student')].index,inplace=True)
    tr_data.drop(tr_data[(tr_data['Work Pressure'].isnull()) & (tr_data['Working Professional or Student']!='Student')].index,inplace=True)
    tr_data.drop(tr_data[(tr_data['CGPA'].isnull()) & (tr_data['Working Professional or Student']=='Student')].index,inplace=True)
    tr_data.drop(tr_data[(tr_data['Study Satisfaction'].isnull()) & (tr_data['Working Professional or Student']=='Student')].index,inplace=True)
    tr_data= self.clean_data(tr_data)
    self.df= self.clean_data(self.df)

    # Initialize Label encoder
    tr_data['Degree']=tr_data['Degree'].map({'Other' :0,'Class 12':1, "Bachelors" : 2, "Masters":3, "Doctorate":4})
    tr_data['Dietary Habits']=tr_data['Dietary Habits'].map({'Other':0, 'Healthy':3, 'Unhealthy':1, 'Moderate':2})

    self.df['Degree']=self.df['Degree'].map({'Other' :0,'Class 12':1, "Bachelors" : 2, "Masters":3, "Doctorate":4})
    self.df['Dietary Habits']=self.df['Dietary Habits'].map({'Other':0, 'Healthy':3, 'Unhealthy':1, 'Moderate':2})

    le = LabelEncoder()
    for i in tr_data.select_dtypes(include="object").columns.to_list():
      tr_data[i] = le.fit_transform(tr_data[i])
      self.df[i] = le.transform(self.df[i])

    num_col=['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
    # MinMax scaling of training data
    scaler = MinMaxScaler()
    tr_data[num_col] = scaler.fit_transform(tr_data[num_col])
    self.df[num_col] = scaler.transform(self.df[num_col])

    return self.df
    

st.subheader("**Mental Health Prediction**")
try:
  pid = st.text_input("Enter your ID")
  name = st.text_input("Enter your Name")
  gen = st.selectbox("Select your Gender",('Male','Female'))
  age = st.text_input("Enter your Age")
  city = st.selectbox("Select your City",('Kalyan', 'Patna', 'Vasai-Virar', 'Kolkata', 'Meerut', 'Ahmedabad',
                                'Visakhapatnam', 'Pune', 'Ludhiana', 'Rajkot', 'Srinagar', 'Mumbai',
                                'Indore', 'Surat', 'Varanasi', 'Agra', 'Hyderabad', 'Jaipur', 'Kanpur',
                                'Vadodara', 'Lucknow', 'Nagpur', 'Thane', 'Bangalore', 'Chennai', 'Ghaziabad',
                                'Delhi', 'Bhopal', 'Faridabad', 'Nashik', 'Other'))
  role = st.selectbox("Select your role", ("Working Professional","Student"))
  prof = st.selectbox("Select your Profession",('Student', 'Teacher', 'Content Writer', 'Architect', 'Consultant',
                                              'HR Manager', 'Pharmacist', 'Doctor', 'Business Analyst',
                                              'Entrepreneur', 'Chemist', 'Financial Analyst', 'Chef',
                                              'Educational Consultant', 'Data Scientist', 'Researcher', 'Lawyer',
                                              'Customer Support', 'Marketing Manager', 'Pilot', 'Travel Consultant',
                                              'Plumber', 'Sales Executive', 'Manager', 'Judge', 'Electrician',
                                              'Software Engineer', 'Civil Engineer', 'UX/UI:Designer', 'Digital Marketer',
                                              'Accountant', 'Mechanical Engineer', 'Graphic Designer', 'Research Analyst',
                                              'Other'))
  acpres= st.selectbox("Rate your Academic Pressure",(0,1,2,3,4,5))
  wkpres= st.selectbox("Rate your Work Pressure",(0,1,2,3,4,5))
  cgpa = st.text_input("Enter your CGPA")
  ssat = st.selectbox("Rate your Study Satisfaction",(0,1,2,3,4,5))
  jsat = st.selectbox("Rate your Job Satisfaction",(0,1,2,3,4,5))
  sleepdur = st.selectbox("Select your Sleep Duration",('Less than 5 hours','7-8 hours','More than 8 hours','5-6 hours','Other'))
  diet = st.selectbox("Select your Dietary Habits",('Healthy', 'Unhealthy', 'Moderate'))
  deg = st.text_input("Enter your Degree")
  suicide = st.selectbox("Have you ever had suicidal thoughts ?",('No','Yes'))
  hrs = st.text_input("Enter your Work/Study Hours")
  fstress = st.selectbox("Rate your Financial Stress",(0,1,2,3,4,5))
  fhis = st.selectbox("Family History of Mental Illness",('No','Yes'))

  if 'clicked' not in st.session_state:
      st.session_state.clicked = False

  def click_button():
      st.session_state.clicked = True

  st.button('Submit', on_click=click_button)

  if st.session_state.clicked:
    
    data=pd.DataFrame({'id' : [pid],'Name' : [name],'Gender': [gen],'Age' : [age],
    'City' : [city], 'Working Professional or Student' : [role],'Profession' : [prof],
    'Academic Pressure': [acpres],'Work Pressure': [wkpres],'CGPA': [cgpa],
    'Study Satisfaction':[ssat],'Job Satisfaction': [jsat],'Sleep Duration': [sleepdur],
    'Dietary Habits': [diet],'Degree':[deg],"Have you ever had suicidal thoughts ?": [suicide],
    'Work/Study Hours': [hrs],'Financial Stress': [fstress],'Family History of Mental Illness' : [fhis]})

    pred = Prediction(data)
    data_c = pred.Preprocess()

    s_col = ['Age', 'Study Satisfaction', 'CGPA', 'Academic Pressure', 'Have you ever had suicidal thoughts ?',
    'Job Satisfaction', 'Work Pressure', 'Working Professional or Student', 'Financial Stress', 'Work/Study Hours',
      'Degree', 'Profession', 'Dietary Habits']
    
    data_c = data_c[s_col]
    new_sample_tensor = tf.convert_to_tensor(data_c.values, dtype=tf.float32) 
    best_model= tf.keras.models.load_model('/content/drive/MyDrive/Project-5-Predict-depression/model_rf.keras')
    ts_predict= best_model.predict(new_sample_tensor)
    ts_pred = ts_predict.round().astype(int).flatten()[0]
    if ts_pred==0:
      st.subheader(f"**Great {name}!!!, you are Healthy and mentally stable**")
    else:
      st.subheader(f"**{name}, No matter how hard things are, Standup and Win over the battle of Depression to enjoy your beautiful life**")

  else:
      st.write('Kindly click Submit button to predict your Mental Health!')

except:
  st.write('After changing the details,Kindly click Submit button to predict your Mental Health!')
