import streamlit as st
import json
#import torch
from collections import Counter
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler 
import category_encoders as one_hot
from sklearn.linear_model import LogisticRegression



#================================================#
#        Loads Model and ready the pipeline      #
#================================================#

loaded_scale = joblib.load('./scale.pkl')
loaded_ohe = joblib.load('./ohe.pkl')
loaded_model = joblib.load('./titanic_baseline_model.pkl')

@st.cache
def preparation_predict(input_prediction):
    test_df = pd.DataFrame({'Pclass': [input_prediction[0]],
                            'Name': [input_prediction[1]],
                            'Sex' : [input_prediction[2]],
                            'Age': [input_prediction[3]],
                            'SibSp': [input_prediction[4]],
                            'Parch': [input_prediction[5]],
                            'Fare':[input_prediction[6]],
                            'Embarked':[input_prediction[7]]
                           })
    new_input_pred = loaded_ohe.transform(test_df)
    new_scale = loaded_scale.transform(new_input_pred)
    result = loaded_model.predict(new_scale)[0]

    if result == 0:
        return 'Not Survived'
    else:
        return 'Survived'




#===========================================#
#              Streamlit Code               #
#===========================================#
desc = "Check out the approach solution code [here](https://www.kaggle.com/code/davious/titanic-solution-baseline-model)!"

st.markdown('<h1 style="text-align: center;font-family: Helvetica, sans-serif;color: #66FCF1;">TITANIC SURVIVAL PREDICTOR</h1>',unsafe_allow_html=True)
#st.title('Titanic Survival Predictor')
st.image("ship_titanic.png")
st.write(desc)
st.write('')
st.write('')
st.write('')
st.write('')
col1, col2 = st.columns(2)

with col1:
    title = st.selectbox(
     'Title',
     ('Mr', 'Mrs', 'Miss','Ms','Master','Mlle','Rev','Jonkheer','Mme',
     'Major', 'Capt', 'Don','Col','No Title'))


with col2:
    user_name = st.text_input('Full Name')


st.write(f'Your Name: {title} {user_name}')
st.write('')
st.write('')



col_sex, col_age = st.columns(2)

with col_sex:
    sex = st.selectbox('Sex', ('Male','Female'))
    if sex == 'Male':
        sex = 0
    elif sex == 'Female':
        sex = 1

    st.write('')
    st.write('')
    sibsp = st.number_input('# of siblings / spouses aboard the Titanic', min_value = 0)
    sibsp = int(sibsp)
    st.write('')
    st.write('')

with col_age:
    age = st.number_input('Age', min_value = 0.0)
    if age > 1:
        age = int(age)
    st.write('')
    st.write('')
    parch = st.number_input('# of parents / children aboard the Titanic', min_value = 0)
    parch = int(parch)
    st.write('')
    st.write('')


col_pclass, col_fare, col_ticket = st.columns(3)

with col_pclass:
    pclass = st.selectbox('Ticket class', ('1st Class','2nd Class', '3rd Class'))
    if pclass == '1st Class':
        pclass = 1
    if pclass == '2nd Class':
        pclass = 2
    if pclass == '3rd Class':
        pclass = 3
    st.write('')
    st.write('')

with col_fare:
    fare = st.number_input('Ticket Price', min_value = 0.0, value = 1000.0)
    st.write('')
    st.write('')

with col_ticket:
    ticket = st.text_input('Ticket Number')
    st.write('')
    st.write('')

col_cabin, col_embark = st.columns(2)

with col_cabin:
    cabin = st.text_input('Cabin Number')
    st.write('')
    st.write('')

with col_embark:
    embarked = st.selectbox('Port Embarked', ('Cherbourg','Queenstown', 'Southampton'))
    if embarked == 'Cherbourg':
        embarked = 'C'
    elif embarked == 'Queenstown':
        embarked = 'Q'
    elif embarked == 'Southampton':
        embarked = 'S'



if st.button('Predict'):
    input_predict = [pclass,title,sex,age,sibsp,parch,fare,embarked]
    result_model = preparation_predict(input_predict)
    #st.write(result_model)

    if result_model == "Not Survived":
        st.markdown('<h3 style="border-style:solid; text-align: center;font-family: Helvetica, sans-serif;color: #FF2400;">Not Survived</h3>',unsafe_allow_html=True)
    else:
        st.markdown('<h3 style="border-style:solid;text-align: center;font-family: Helvetica, sans-serif;color: #86DC3D;">Survived</h3>',unsafe_allow_html=True)

