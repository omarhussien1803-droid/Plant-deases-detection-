
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Maqaawwan dhibee Afaan Oromootiin
class_names = {
    0: "Apple: Scab (Madaa Jirmaa)", 1: "Apple: Black Rot (Cunfuree Gurraacha)", 
    2: "Apple: Cedar Rust (Waraantee)", 3: "Apple: Fayyaa",
    4: "Blueberry: Fayyaa", 5: "Cherry: Powdery Mildew (Aara Adii)", 6: "Cherry: Fayyaa",
    7: "Boqqoolloo: Cercospora Leaf Spot (Madaa Baalaa)", 8: "Boqqoolloo: Common Rust (Waraantee)", 
    9: "Boqqoolloo: Northern Leaf Blight (Cunfuree)", 10: "Boqqoolloo: Fayyaa", 
    11: "Grape: Black Rot (Cunfuree Wayinii)", 12: "Grape: Black Measles",
    13: "Grape: Leaf Blight (Cunfuree Baala Wayinii)", 14: "Grape: Fayyaa", 
    15: "Orange: Haunglongbing (Dhibee bifa keelloo)",
    16: "Peach: Bacterial Spot (Madaa Bakteeriyaa)", 17: "Peach: Fayyaa", 
    18: "Pepper Bell: Bacterial Spot (Madaa Bakteeriyaa)", 19: "Pepper Bell: Fayyaa", 
    20: "Dinicha: Early Blight (Cunfuree Ganamaa)", 21: "Dinicha: Late Blight (Cunfuree Galgalaa)",
    22: "Dinicha: Fayyaa", 23: "Raspberry: Fayyaa", 24: "Soybean: Fayyaa",
    25: "Squash: Powdery Mildew (Aara Adii)", 26: "Strawberry: Leaf Scorch (Gubata Baalaa)", 
    27: "Strawberry: Fayyaa",
    28: "Tumaatima: Bacterial Spot (Madaa Bakteeriyaa)", 29: "Tumaatima: Early Blight (Cunfuree Ganamaa)", 
    30: "Tumaatima: Late Blight (Cunfuree Galgalaa)", 31: "Tumaatima: Leaf Mold (Aara Baalaa)", 
    32: "Tumaatima: Septoria Leaf Spot", 33: "Tumaatima: Spider Mites (Ilbiisa)",
    34: "Tumaatima: Target Spot", 35: "Tumaatima: Yellow Leaf Curl Virus (Virus Baala Maru)", 
    36: "Tumaatima: Mosaic Virus", 37: "Tumaatima: Fayyaa"
}

st.title("AI Dhibee Biqiltootaa Addaan Baasu")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mira_plant_model.h5')

model = load_my_model()

file = st.file_uploader("Suuraa baala biqiltuu fe'i", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert('RGB').resize((224, 224))
    st.image(img, caption="Suuraa qoratamanif dhiyaate")
    
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    
    prediction = model.predict(x)
    label = np.argmax(prediction)
    maqaa = class_names.get(label, "Dhibee hin beekamne")
    
    st.success(f"AI'n kee akka jedhutti: **{maqaa}**")
