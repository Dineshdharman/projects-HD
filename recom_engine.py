import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

# Set environment variable to avoid UnicodeEncodeError
os.environ["PYTHONIOENCODING"] = "utf-8"

# Load precomputed feature vectors and filenames
try:
    feature_list = np.array(pickle.load(open(r'C:\Users\HP\Desktop\fashion[1]\Fashion_Recommander_System-main\featurevector.pkl', 'rb')))
    filenames = pickle.load(open(r'C:\Users\HP\Desktop\fashion[1]\Fashion_Recommander_System-main\filenames.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading precomputed data: {e}")
    st.stop()

# Load the pre-trained ResNet50 model and modify it
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Man & Women Fashion Recommender System')

# Ensure the uploads directory exists
uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    try:
        os.makedirs(uploads_dir)
    except Exception as e:
        st.error(f"Error creating uploads directory: {e}")
        st.stop()

def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload and recommendation
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    # Check if the uploaded file is an image
    if uploaded_file.type.startswith('image/'):
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            # Display the uploaded image
            display_image = Image.open(file_path)
            resized_img = display_image.resize((200, 200))
            st.image(resized_img, caption='Uploaded Image')

            # Extract features and get recommendations
            features = extract_feature(file_path, model)
            indices = recommend(features, feature_list)

            # Display the recommended images
            st.write("Recommended items:")
            cols = st.columns(10)
            for i, col in enumerate(cols):
                if i < len(indices[0]):
                    with col:
                        st.image(filenames[indices[0][i]])
        else:
            st.error("Failed to save the uploaded file.")
    else:
        st.error("Please upload a valid image file.")
else:
    st.info("Please upload an image to get recommendations.")