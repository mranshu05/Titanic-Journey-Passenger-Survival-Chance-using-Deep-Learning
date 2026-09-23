import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle


st.title("Passenger survival chance in the Titanic Journey")

pclass=st.slider("Enter the Passenger class for the user",1,3)

sex=st.selectbox("Enter the Passenger Gender",['male','female'])
sibsp=st.slider("Enter the Passenger's total number of Siblings and Spouse",1,8)
parch=st.slider("Enter the Passenger's total number of Parents and Child",0,6)
fare=st.number_input("Enter the Fare of the Passenger")
embarked=st.selectbox("Select the Passenger's station from where he/she started the Journey",['Southhampton','Chebourg','Queenstown'])

data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])

model=load_model('my_model.keras')

with open('label_encoder.pkl','rb') as file:
    label=pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehot=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

data['Sex']=label.transform(data['Sex'])
embarked=onehot.transform(data[['Embarked']])

embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())

data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)

data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

Y=model.predict(data)
Y=Y[0][0]

def Chance(Y):
    if Y>0.5:
        return "The Passenger will survive the Journey"
    else:
        return "The Passenger will not survive the Journey"

if st.button('Predict survival chance'):
    st.write('Probability of Passenger survival chance',Y)
    st.write(Chance(Y))




