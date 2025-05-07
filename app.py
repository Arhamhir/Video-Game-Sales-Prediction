import streamlit as st
import joblib as jb
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
model = jb.load('mymodel.pkl')
scaler = jb.load('myscaler.pkl')
dictionary = jb.load('mydictionary.pkl')

st.header('Video Games Global Sales Prediction')

platform = st.text_input("Platform Name")
if platform:
    if platform.upper() not in dictionary['Platform']:
        st.error('Platform not supported')
release = st.slider('Year of release', 1980, 2025, 2005)
genre = st.selectbox('Genre', dictionary['Genre'])
publisher = st.text_input('Publisher')
if publisher:
    if publisher.upper() not in dictionary['Publisher']:
        st.error("Publisher not found")
copies_na = st.number_input('NA sales (Copies sold in millions)', 0.0, 3.5, 2.0)


if st.button('Predict'):
    platform_encoded = dictionary['Platform'].index(platform.upper())
    genre_encoded = dictionary['Genre'].index(genre.upper())
    publisher_encoded = dictionary['Publisher'].index(publisher.upper())

    user_in = [[platform_encoded, release, genre_encoded, publisher_encoded, copies_na]]
    scaled = scaler.transform(user_in)
    prediction = model.predict(scaled)[0]
    st.success(f"Estimated copies to be sold globally: {prediction * 1_000_000:.0f}")

