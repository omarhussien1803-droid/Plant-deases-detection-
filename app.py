
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Maqaawwan dhibee (Akka dataset keetti sirreessi)
class_names = {
    0: "Baala Fayyaa",
    1: "Cunfuree (Blight)",
    2: "Waraantee (Rust)",
    3: "Dhibee 3ffaa",
    4: "Dhibee 4ffaa",
    5: "Dhibee 5ffaa",
    6: "Dhibee 6ffaa",
    7: "Dhibee Mashoo (Powdery Mildew)"
}

# Maqaa weebsaayitii
st.title("AI Dhibee Biqiltootaa Addaan Baasu")

# Model kee dubbisuu
@st.cache_resource
def load_my_model():
    # Faayila .h5 GitHub irratti fe'ame sana dubbisa
    return tf.keras.models.load_model('mira_plant_model.himport streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Maqaawwan dhibee hunda (PlantVillage standard 38 labels)
class_names = {
    0: "Apple Scab", 1: "Apple Black Rot", 2: "Apple Cedar Rust", 3: "Apple Healthy",
    4: "Blueberry Healthy", 5: "Cherry Powdery Mildew", 6: "Cherry Healthy",
    7: "Corn Cercospora Leaf Spot", 8: "Corn Common Rust", 9: "Corn Northern Leaf Blight",
    10: "Corn Healthy", 11: "Grape Black Rot", 12: "Grape Black Measles",
    13: "Grape Leaf Blight", 14: "Grape Healthy", 15: "Orange Haunglongbing",
    16: "Peach Bacterial Spot", 17: "Peach Healthy", 18: "Pepper Bell Bacterial Spot",
    19: "Pepper Bell Healthy", 20: "Potato Early Blight", 21: "Potato Late Blight",
    22: "Potato Healthy", 23: "Raspberry Healthy", 24: "Soybean Healthy",
    25: "Squash Powdery Mildew", 26: "Strawberry Leaf Scorch", 27: "Strawberry Healthy",
    28: "Tomato Bacterial Spot", 29: "Tomato Early Blight", 30: "Tomato Late Blight",
    31: "Tomato Leaf Mold", 32: "Tomato Septoria Leaf Spot", 33: "Tomato Spider Mites",
    34: "Tomato Target Spot", 35: "Tomato Yellow Leaf Curl Virus", 36: "Tomato Mosaic Virus",
    37: "Tomato Healthy"
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
    
    # Model kee outputs dhibee qofa yoo ta'e
    label = np.argmax(prediction)
    maqaa = class_names.get(label, "Dhibee hin beekamne")
    
    st.success(f"AI'n kee akka jedhutti: **{maqaa}**")
    
