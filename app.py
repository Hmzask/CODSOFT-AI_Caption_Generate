import streamlit as st
from models.caption_model import ImageCaptioningModel
from utils.image_utils import load_image

@st.cache_resource # It caches the model upload. To Avoid uploading the model on every startWrite a Python program to find and remove duplicate rows.
.
def load_model():
    return ImageCaptioningModel()

model = load_model()

st.set_page_config(page_title="Image Captioning AI (BLIP-2)", layout="wide")
st.title("🖼️ Image Captioning AI using BLIP-2")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = load_image(uploaded_image)
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Generate Caption"):
        with st.spinner("Analyzing image..."):
            caption = model.generate_caption(image)

        st.success("Caption Generated!")
        st.write(f"### 📝 Caption: **{caption}**")

