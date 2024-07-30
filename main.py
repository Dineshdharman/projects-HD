import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

feature_list = np.array(pickle.load(open('featurevector.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Man & Women Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def extract_feature(img_path, model):
    img=cv2.imread(img_path)
    img=cv2.resize(img, (224,224))
    img=np.array(img)
    expand_img=np.expand_dims(img, axis=0)
    pre_img=preprocess_input(expand_img)
    result=model.predict(pre_img).flatten()
    normalized=result/norm(result)
    return normalized

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
print(uploaded_file)
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        resized_img = display_image.resize((200, 200))
        st.image(resized_img)
        # feature extract
        features = extract_feature(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        col1,col2,col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
        with col6:
            st.image(filenames[indices[0][5]])
        with col7:
            st.image(filenames[indices[0][6]])
        with col8:
            st.image(filenames[indices[0][7]])
        with col9:
            st.image(filenames[indices[0][8]])
        with col10:
            st.image(filenames[indices[0][9]])
    else:
        st.header("Some error occured in file upload")







# import tkinter as tk
# from tkinter import filedialog, messagebox, ttk
# from PIL import Image, ImageTk
# import numpy as np
# import pickle
# import tensorflow as tf
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from sklearn.neighbors import NearestNeighbors
# from numpy.linalg import norm
# import os
# import cv2

# # Load pre-trained model and features
# feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
# filenames = pickle.load(open('filenames.pkl', 'rb'))

# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model.trainable = False
# model = tf.keras.Sequential([
#     model,
#     GlobalMaxPooling2D()
# ])

# def extract_feature(img_path, model):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (224, 224))
#     img = np.array(img)
#     expand_img = np.expand_dims(img, axis=0)
#     pre_img = preprocess_input(expand_img)
#     result = model.predict(pre_img).flatten()
#     normalized = result / norm(result)
#     return normalized

# def recommend(features, feature_list):
#     neighbors = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='euclidean')
#     neighbors.fit(feature_list)
#     distances, indices = neighbors.kneighbors([features])
#     return indices

# class FashionRecommenderApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Man & Women Fashion Recommender System")
#         self.root.geometry("1200x800")

#         self.frames = {}
#         for F in (LandingPage, SignUpPage, LoginPage, UserProfilePage, RecommendationPage, SearchPage, CartPage, WishlistPage, AdminPanel):
#             page_name = F.__name__
#             frame = F(parent=self.root, controller=self)
#             self.frames[page_name] = frame
#             frame.grid(row=0, column=0, sticky="nsew")

#         self.show_frame("LandingPage")

#     def show_frame(self, page_name):
#         frame = self.frames[page_name]
#         frame.tkraise()

# class LandingPage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller

#         label = tk.Label(self, text="Welcome to the Fashion Recommender System", font=("Arial", 24))
#         label.pack(pady=20)

#         description = tk.Label(self, text="Get personalized fashion recommendations based on your style preferences.", font=("Arial", 14))
#         description.pack(pady=10)

#         sign_up_button = tk.Button(self, text="Sign Up", command=lambda: controller.show_frame("SignUpPage"))
#         sign_up_button.pack(pady=10)

#         login_button = tk.Button(self, text="Login", command=lambda: controller.show_frame("LoginPage"))
#         login_button.pack(pady=10)

# class SignUpPage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller

#         label = tk.Label(self, text="Sign Up", font=("Arial", 24))
#         label.pack(pady=20)

#         self.name_entry = tk.Entry(self)
#         self.name_entry.pack(pady=5)
#         self.name_entry.insert(0, "Name")

#         self.email_entry = tk.Entry(self)
#         self.email_entry.pack(pady=5)
#         self.email_entry.insert(0, "Email")

#         self.password_entry = tk.Entry(self, show='*')
#         self.password_entry.pack(pady=5)
#         self.password_entry.insert(0, "Password")

#         sign_up_button = tk.Button(self, text="Sign Up", command=self.sign_up)
#         sign_up_button.pack(pady=20)

#         login_label = tk.Label(self, text="Already have an account? Login", fg="blue", cursor="hand2")
#         login_label.pack(pady=10)
#         login_label.bind("<Button-1>", lambda e: controller.show_frame("LoginPage"))

#     def sign_up(self):
#         name = self.name_entry.get()
#         email = self.email_entry.get()
#         password = self.password_entry.get()
#         # Here, you would normally add the user data to your database
#         messagebox.showinfo("Success", "Account created successfully!")
#         self.controller.show_frame("LoginPage")

# class LoginPage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller

#         label = tk.Label(self, text="Login", font=("Arial", 24))
#         label.pack(pady=20)

#         self.email_entry = tk.Entry(self)
#         self.email_entry.pack(pady=5)
#         self.email_entry.insert(0, "Email")

#         self.password_entry = tk.Entry(self, show='*')
#         self.password_entry.pack(pady=5)
#         self.password_entry.insert(0, "Password")

#         login_button = tk.Button(self, text="Login", command=self.login)
#         login_button.pack(pady=20)

#         sign_up_label = tk.Label(self, text="Don't have an account? Sign Up", fg="blue", cursor="hand2")
#         sign_up_label.pack(pady=10)
#         sign_up_label.bind("<Button-1>", lambda e: controller.show_frame("SignUpPage"))

#     def login(self):
#         email = self.email_entry.get()
#         password = self.password_entry.get()
#         # Here, you would normally check the user credentials
#         messagebox.showinfo("Success", "Logged in successfully!")
#         self.controller.show_frame("UserProfilePage")

# class UserProfilePage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller

#         label = tk.Label(self, text="User Profile", font=("Arial", 24))
#         label.pack(pady=20)

#         self.name_label = tk.Label(self, text="Name: John Doe")
#         self.name_label.pack(pady=5)

#         self.email_label = tk.Label(self, text="Email: john.doe@example.com")
#         self.email_label.pack(pady=5)

#         self.profile_pic = tk.Label(self, text="Profile Picture")
#         self.profile_pic.pack(pady=5)

#         self.preferences_label = tk.Label(self, text="Preferences")
#         self.preferences_label.pack(pady=5)

#         self.past_purchases_label = tk.Label(self, text="Past Purchases")
#         self.past_purchases_label.pack(pady=5)

#         recommend_button = tk.Button(self, text="Get Recommendations", command=lambda: controller.show_frame("RecommendationPage"))
#         recommend_button.pack(pady=20)

#         search_button = tk.Button(self, text="Search for Items", command=lambda: controller.show_frame("SearchPage"))
#         search_button.pack(pady=10)

# class RecommendationPage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller

#         label = tk.Label(self, text="Recommended Items", font=("Arial", 24))
#         label.pack(pady=20)

#         self.recommend_frame = tk.Frame(self)
#         self.recommend_frame.pack(pady=10, fill=tk.BOTH, expand=True)

#         # Add a scrollbar to the recommendations frame
#         self.canvas = tk.Canvas(self.recommend_frame)
#         self.scrollbar = tk.Scrollbar(self.recommend_frame, orient="vertical", command=self.canvas.yview)
#         self.canvas.configure(yscrollcommand=self.scrollbar.set)

#         self.scrollbar.pack(side="right", fill="y")
#         self.canvas.pack(side="left", fill="both", expand=True)
#         self.canvas.create_window((0, 0), window=self.recommend_frame, anchor="nw")

#         self.recommend_frame.bind("<Configure>", self.on_frame_configure)

#         upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
#         upload_button.pack(pady=20)

#         self.uploaded_image_path = None

#     def on_frame_configure(self, event):
#         self.canvas.configure(scrollregion=self.canvas.bbox("all"))

#     def upload_image(self):
#         file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
#         if file_path:
#             self.uploaded_image_path = file_path
#             img = Image.open(file_path)
#             img = img.resize((200, 200), Image.LANCZOS)
#             img = ImageTk.PhotoImage(img)
#             display_label = tk.Label(self, image=img)
#             display_label.image = img
#             display_label.pack(pady=10)
#             self.recommend_fashion()

#     def recommend_fashion(self):
#         if self.uploaded_image_path:
#             features = extract_feature(self.uploaded_image_path, model)
#             indices = recommend(features, feature_list)

#             for widget in self.recommend_frame.winfo_children():
#                 widget.destroy()

#             for idx in range(1, 11):
#                 img_path = filenames[indices[0][idx]]
#                 img = Image.open(img_path)
#                 img = img.resize((200, 200), Image.LANCZOS)
#                 img = ImageTk.PhotoImage(img)

#                 frame = tk.Frame(self.recommend_frame)
#                 frame.pack(side=tk.LEFT, padx=10, pady=10)

#                 img_label = tk.Label(frame, image=img)
#                 img_label.image = img
#                 img_label.pack()

#                 # Add details like name, price, etc. here if available

# class SearchPage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller

#         label = tk.Label(self, text="Search for Items", font=("Arial", 24))
#         label.pack(pady=20)

#         self.search_entry = tk.Entry(self)
#         self.search_entry.pack(pady=5)

#         search_button = tk.Button(self, text="Search", command=self.search_items)
#         search_button.pack(pady=10)

#         self.search_results = tk.Frame(self)
#         self.search_results.pack(pady=10, fill=tk.BOTH, expand=True)

#     def search_items(self):
#         query = self.search_entry.get()
#         # Perform search and display results (dummy implementation for now)
#         for widget in self.search_results.winfo_children():
#             widget.destroy()

#         # Example search result
#         example_image = Image.new("RGB", (200, 200), "blue")
#         example_image = ImageTk.PhotoImage(example_image)
#         example_label = tk.Label(self.search_results, image=example_image)
#         example_label.image = example_image
#         example_label.pack(pady=10)

# class CartPage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller

#         label = tk.Label(self, text="Shopping Cart", font=("Arial", 24))
#         label.pack(pady=20)

#         self.cart_items = tk.Listbox(self, width=50, height=10)
#         self.cart_items.pack(pady=10)

#         # Example items (replace with dynamic content in real application)
#         self.cart_items.insert(tk.END, "Item 1 - $19.99")
#         self.cart_items.insert(tk.END, "Item 2 - $29.99")
#         self.cart_items.insert(tk.END, "Item 3 - $39.99")

#         self.total_price_label = tk.Label(self, text="Total Price: $89.97", font=("Arial", 16))
#         self.total_price_label.pack(pady=10)

#         checkout_button = tk.Button(self, text="Proceed to Checkout", command=self.checkout)
#         checkout_button.pack(pady=20)

#     def checkout(self):
#         messagebox.showinfo("Checkout", "Proceeding to checkout...")

# class WishlistPage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller

#         label = tk.Label(self, text="Wishlist", font=("Arial", 24))
#         label.pack(pady=20)

#         self.wishlist_items = tk.Listbox(self, width=50, height=10)
#         self.wishlist_items.pack(pady=10)

#         # Example items (replace with dynamic content in real application)
#         self.wishlist_items.insert(tk.END, "Wishlist Item 1 - $19.99")
#         self.wishlist_items.insert(tk.END, "Wishlist Item 2 - $29.99")

#         move_to_cart_button = tk.Button(self, text="Move to Cart", command=self.move_to_cart)
#         move_to_cart_button.pack(pady=20)

#     def move_to_cart(self):
#         # Placeholder action for moving items to the cart
#         messagebox.showinfo("Move to Cart", "Items moved to cart!")

# class AdminPanel(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller

#         label = tk.Label(self, text="Admin Panel", font=("Arial", 24))
#         label.pack(pady=20)

#         user_management_button = tk.Button(self, text="Manage Users")
#         user_management_button.pack(pady=10)

#         product_management_button = tk.Button(self, text="Manage Products")
#         product_management_button.pack(pady=10)

#         analytics_button = tk.Button(self, text="View Analytics")
#         analytics_button.pack(pady=10)

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = FashionRecommenderApp(root)
#     root.mainloop()
