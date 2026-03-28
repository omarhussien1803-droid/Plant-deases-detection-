import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Maqaa weebsaayitii
st.title("AI Dhibee Biqiltootaa Addaan Baasu")

# Model kee isa GitHub irra jiru dubbisuu
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mira_plant_model.h5')

model = load_my_model()

# Suuraa ittiin qoruu
file = st.file_uploader("Suuraa baala biqiltuu fe'i", type=["jpg", "png"])

if file:
    img = Image.open(file).resize((224, 224))
    st.image(img, caption="Suuraa qoratamaniif dhiyaate")
    
    # AI'n akka tilmaamu gochuu
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    
    st.success(f"AI'n kee akka jedhutti koodiin dhibee: {np.argmax(prediction)}")
