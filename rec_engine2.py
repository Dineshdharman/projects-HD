import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess_input, decode_predictions
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

# Set environment variable to avoid UnicodeEncodeError
os.environ["PYTHONIOENCODING"] = "utf-8"

# Define paths
base_dir = r'C:\Users\HP\Desktop\fashion[1]\Fashion_Recommander_System-main'
feature_vector_path = os.path.join(base_dir, 'featurevector.pkl')
filenames_path = os.path.join(base_dir, 'filenames.pkl')
uploads_dir = os.path.join(base_dir, 'uploads')

# Load precomputed feature vectors and filenames
try:
    feature_list = np.array(pickle.load(open(feature_vector_path, 'rb')))
    filenames = pickle.load(open(filenames_path, 'rb'))
    st.write("Filenames loaded:", filenames)  # Debug: Print filenames
except Exception as e:
    st.error(f"Error loading precomputed data: {e}")
    st.stop()

# Load the pre-trained ResNet50 model and modify it
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_model.trainable = False
resnet_model = tf.keras.Sequential([
    resnet_model,
    GlobalMaxPooling2D()
])

# Load the MobileNetV2 model for image labeling
mobilenet_model = MobileNetV2(weights='imagenet')

st.title('Man & Women Fashion Recommender System')

# Ensure the uploads directory exists
if not os.path.exists(uploads_dir):
    try:
        os.makedirs(uploads_dir)
    except Exception as e:
        st.error(f"Error creating uploads directory: {e}")
        st.stop()

def resolve_path(relative_path):
    """ Resolve and return the absolute path from the relative path. """
    return os.path.join(base_dir, relative_path)

def save_uploaded_file(uploaded_file):
    """ Save uploaded file to the uploads directory. """
    try:
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        st.write(f"Saving file to: {file_path}")  # Debug: Print path where file is saved
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def extract_feature(img_path, model):
    """ Extract features from an image using the pre-trained model. """
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image not found or unable to read.")
        img = cv2.resize(img, (224, 224))
        img = np.array(img)
        expand_img = np.expand_dims(img, axis=0)
        pre_img = preprocess_input(expand_img)
        result = model.predict(pre_img).flatten()
        normalized = result / norm(result)
        return normalized
    except Exception as e:
        st.error(f"Error processing image at '{img_path}': {e}")
        return None

def label_image(img_path, model):
    """ Label an image using the pre-trained MobileNetV2 model. """
    try:
        st.write(f"Attempting to label image at: {img_path}")  # Debug: Print image path
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Image not found or unable to read.")
        img = cv2.resize(img, (224, 224))
        img = np.array(img)
        expand_img = np.expand_dims(img, axis=0)
        pre_img = mobilenet_preprocess_input(expand_img)
        preds = model.predict(pre_img)
        label = decode_predictions(preds, top=1)[0][0][1]  # Get the most likely label
        return label
    except Exception as e:
        st.error(f"Error labeling image at '{img_path}': {e}")
        return None

def recommend(features, feature_list):
    """ Find nearest neighbors based on extracted features. """
    try:
        neighbors = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        
        # Debugging information
        st.write(f"Indices of recommendations: {indices}")
        st.write(f"Distances of recommendations: {distances}")
        
        return indices
    except Exception as e:
        st.error(f"Error finding recommendations: {e}")
        return None

def resolve_path(relative_path):
    """Resolve and return the absolute path from the relative path."""
    return os.path.join(base_dir, relative_path.replace('/', os.path.sep))

def display_recommendations(indices, filenames):
    st.write("Recommended items:")
    cols = st.columns(10)
    for i, col in enumerate(cols):
        if i < len(indices[0]):
            try:
                image_path = resolve_path(filenames[indices[0][i]])
                st.write(f"Checking: {image_path}")  # Debug: Print path being checked
                if os.path.exists(image_path):
                    with col:
                        st.image(image_path)
                else:
                    with col:
                        st.error(f"Image not found: {image_path}")
            except Exception as e:
                st.error(f"Error displaying image: {e}")

def filter_images_by_label(label, filenames, labels_dict):
    """ Filter filenames by the given label. """
    filtered_filenames = [filename for filename in filenames if labels_dict.get(os.path.basename(filename), '').lower() == label.lower()]
    st.write(f"Filtering for label '{label}': Found {len(filtered_filenames)} matching images.")
    return filtered_filenames

# Automatically label images using MobileNetV2
image_labels = {}
for filename in filenames:
    image_path = resolve_path(filename)
    st.write(f"Processing image: {image_path}")  # Debug: Print image path being processed
    label = label_image(image_path, mobilenet_model)
    if label:
        image_labels[filename] = label

# Text input for product name
product_name = st.text_input("Enter product name (e.g., tshirt, shoes):")

if product_name:
    filtered_filenames = filter_images_by_label(product_name, filenames, image_labels)
    
    if filtered_filenames:
        st.write(f"Found {len(filtered_filenames)} images for '{product_name}'")
        
        # Display filtered images
        cols = st.columns(5)
        for i, filename in enumerate(filtered_filenames):
            if i < len(filtered_filenames):
                try:
                    image_path = resolve_path(filename)
                    st.write(f"Checking: {image_path}")  # Debug: Print path being checked
                    if os.path.exists(image_path):
                        with cols[i % 5]:
                            st.image(image_path)
                    else:
                        with cols[i % 5]:
                            st.error(f"Image not found: {image_path}")
                except Exception as e:
                    st.error(f"Error displaying image: {e}")
    else:
        st.warning(f"No images found for '{product_name}'")
else:
    st.info("Please enter a product name to get recommendations.")
