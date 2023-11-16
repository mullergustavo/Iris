# https://irisdata.streamlit.app

import streamlit as st
from PIL import Image
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#config
st.set_page_config(page_title="Iris Dataset | +Predictor", page_icon="🌼")

iris_measurement = Image.open('iris_measurement.jpg')
iris_setosa = Image.open('iris_setosa.jpg')
iris_versicolor = Image.open('iris_versicolor.jpg')
iris_virginica = Image.open('iris_virginica.jpg')

con = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Iris", con)
con.close()

st.title("Iris Dataset")
st.write("The iris dataset is one of the most popular datasets for evaluating classification methods. It is a 3 class dataset with 50 instances each, donated by Ronald Fisher in 1936. The data used by this app was obtained from [Kaggle](https://www.kaggle.com/datasets/uciml/iris).")
st.write("This web application loads the dataset from a .sqlite file, transforms it into a dataframe with pandas (for easy visualization and manipulation), in order to apply the Support Vector Machines model, and predict what class of iris you have from 4 numerical input data.")

df = df.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], ['Iris setosa', 'Iris versicolor','Iris virginica'])

if st.checkbox('Display interactive dataframe'):
    st.dataframe(df, use_container_width=True)

st.divider()

X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=23)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

st.title("Iris Dataset | Predictor")

st.image(iris_measurement)

st.write("Insert the required data to run the iris specie prediction")
inp_SepalLengthCm = st.number_input('Insert a number for the Sepal Length in Centimeters', value=6.3, help='For example: 6.9')
inp_SepalWidthCm = st.number_input('Insert a number for the Sepal Width in Centimeters', value=2.3, help='For example: 3.1')
inp_PetalLengthCm = st.number_input('Insert a number for the Petal Length in Centimeters', value=4.4, help='For example: 5.4')
inp_PetalWidthCm = st.number_input('Insert a number for the Petal Width in Centimeters', value=1.3, help='For example: 2.1')

user_prediction = svm.predict([[inp_SepalLengthCm, inp_SepalWidthCm, inp_PetalLengthCm, inp_PetalWidthCm]])

if user_prediction == ['Iris setosa']:
    st.write('The result of the specie prediction is: Iris setosa')
    st.image(iris_setosa, caption='Iris setosa')
elif user_prediction == ['Iris versicolor']:
    st.write('The result of the specie prediction is: Iris versicolor')
    st.image(iris_versicolor, caption='Iris versicolor')
else:
    st.write('The result of the specie prediction is: Iris virginica')
    st.image(iris_virginica, caption='Iris virginica')

st.divider()

user_option = st.selectbox('Select the specie of iris you want to explore in the dataset: ', df['Species'].unique())
st.write('Your selection: ', user_option)

st.dataframe(df.loc[df['Species'] == user_option], use_container_width=True)
st.divider()

st.write("Do you have questions, suggestions, or want to contact me? Visit my profile on [LinkedIn](https://www.linkedin.com/in/gustavomuller) or [GitHub](https://github.com/mullergustavo)")
