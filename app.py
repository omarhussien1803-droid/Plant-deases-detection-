
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
    return tf.keras.models.load_model('mira_plant_model.h5')

model = load_my_model()

# Suuraa ittiin qoruu
file = st.file_uploader("Suuraa baala biqiltuu fe'i", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert('RGB').resize((224, 224))
    st.image(img, caption="Suuraa qoratamanif dhiyaate")

    # AI'n akka tilmaamu gochuu
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    label = np.argmax(prediction)
    
    # Maqaa dhibee baasuu
    maqaa = class_names.get(label, "Dhibee hin beekamne")
    
    st.success(f"AI'n kee akka jedhutti: **{maqaa}**")
