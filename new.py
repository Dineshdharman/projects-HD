'''import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

# Load pre-trained models and data
feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
  
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image
def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

# Function to recommend images based on extracted features
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# Function to handle file upload and processing
def upload_image():
    uploaded_file = filedialog.askopenfile(title='Choose an image', filetypes=[('Image files', '*.jpg; *.jpeg; *.png')])
    if uploaded_file:
        try:
            # Display uploaded image
            img = Image.open(uploaded_file.name)
            img = img.resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk  # keep a reference to the image

            # Extract features from the uploaded image
            features = extract_feature(uploaded_file.name, model)

            # Recommend images based on extracted features
            indices = recommend(features, feature_list)

            # Display recommended images
            display_recommendations(indices)

        except Exception as e:
            messagebox.showerror('Error', f'Error uploading image: {str(e)}')

# Function to display recommended images
def display_recommendations(indices):
    for i in range(min(len(indices[0]), 10)):  # Display up to 10 recommendations
        img_path = filenames[indices[0][i]]
        img = Image.open(img_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        recommendation_labels[i].config(image=img_tk)
        recommendation_labels[i].image = img_tk  # keep a reference to the image

# Initialize Tkinter
root = tk.Tk()
root.title('Fashion Recommender System')

# Upload Button
upload_button = tk.Button(root, text='Upload Image', command=upload_image)
upload_button.pack(pady=10)

# Uploaded Image Display
image_label = tk.Label(root)
image_label.pack(pady=10)

# Recommended Images Display
recommendation_labels = []
for i in range(10):  # Display 10 recommendations in a row
    recommendation_label = tk.Label(root)
    recommendation_label.pack(side=tk.LEFT, padx=5)
    recommendation_labels.append(recommendation_label)

root.mainloop()

'''
'''
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

# Load pre-trained models and data
feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image
def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

# Function to recommend images based on extracted features
def recommend(features, feature_list, filters=None):
    neighbors = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    filtered_indices = indices

    if filters:
        filtered_indices = []
        for idx in indices[0]:
            if filters.get('color') and filters['color'] not in filenames[idx]:
                continue
            if filters.get('brand') and filters['brand'] not in filenames[idx]:
                continue
            if filters.get('size') and filters['size'] not in filenames[idx]:
                continue
            if filters.get('material') and filters['material'] not in filenames[idx]:
                continue
            if filters.get('style') and filters['style'] not in filenames[idx]:
                continue
            if filters.get('occasion') and filters['occasion'] not in filenames[idx]:
                continue
            if filters.get('season') and filters['season'] not in filenames[idx]:
                continue
            if filters.get('pattern') and filters['pattern'] not in filenames[idx]:
                continue
            if filters.get('price_range'):
                price = int(filenames[idx].split('_')[-1].replace('.jpg', ''))  # Placeholder for actual price parsing
                if not (filters['price_range'][0] <= price <= filters['price_range'][1]):
                    continue
            filtered_indices.append(idx)
        filtered_indices = [filtered_indices]

    return filtered_indices

class FashionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Fashion Recommender System')
        self.geometry('800x600')

        self.frames = {}
        for F in (HomePage, UserInputPage, ImageUploadPage, RecommendationPage, WishlistPage):
            page_name = F.__name__
            frame = F(parent=self, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame('HomePage')

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        label = tk.Label(self, text="Welcome to Fashion Recommender System", font=('Helvetica', 18, 'bold'))
        label.pack(pady=20)
        button1 = tk.Button(self, text="Start", command=lambda: controller.show_frame('UserInputPage'))
        button1.pack()

class UserInputPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="User Information", font=('Helvetica', 14, 'bold'))
        label.pack(pady=10)

        self.name_label = tk.Label(self, text="Name:")
        self.name_label.pack()
        self.name_entry = tk.Entry(self)
        self.name_entry.pack()

        self.age_label = tk.Label(self, text="Age:")
        self.age_label.pack()
        self.age_entry = tk.Entry(self)
        self.age_entry.pack()

        self.gender_label = tk.Label(self, text="Gender:")
        self.gender_label.pack()
        self.gender_var = tk.StringVar()
        self.gender_menu = ttk.Combobox(self, textvariable=self.gender_var, values=["Male", "Female", "Other"])
        self.gender_menu.pack()

        self.style_label = tk.Label(self, text="Style Preferences:")
        self.style_label.pack()
        self.style_entry = tk.Entry(self)
        self.style_entry.pack()

        self.occasion_label = tk.Label(self, text="Occasion:")
        self.occasion_label.pack()
        self.occasion_var = tk.StringVar()
        self.occasion_menu = ttk.Combobox(self, textvariable=self.occasion_var, values=["Casual", "Formal", "Party", "Office Wear"])
        self.occasion_menu.pack()

        next_button = tk.Button(self, text="Next", command=lambda: controller.show_frame('ImageUploadPage'))
        next_button.pack()

class ImageUploadPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Upload Image", font=('Helvetica', 14, 'bold'))
        label.pack(pady=10)

        self.upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        next_button = tk.Button(self, text="Get Recommendations", command=lambda: controller.show_frame('RecommendationPage'))
        next_button.pack()

    def upload_image(self):
        uploaded_file = filedialog.askopenfile(title='Choose an image', filetypes=[('Image files', '*.jpg; *.jpeg; *.png')])
        if uploaded_file:
            try:
                img = Image.open(uploaded_file.name)
                img = img.resize((200, 200))
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk

                self.features = extract_feature(uploaded_file.name, model)
                self.controller.features = self.features

            except Exception as e:
                messagebox.showerror('Error', f'Error uploading image: {str(e)}')

class RecommendationPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Recommendations", font=('Helvetica', 14, 'bold'))
        label.pack(pady=10)

        # Filters
        self.filter_frame = tk.Frame(self)
        self.filter_frame.pack(pady=10)

        tk.Label(self.filter_frame, text="Color:").grid(row=0, column=0, padx=5)
        self.color_var = tk.StringVar()
        self.color_menu = ttk.Combobox(self.filter_frame, textvariable=self.color_var, values=["Red", "Blue", "Green", "Black", "White"])
        self.color_menu.grid(row=0, column=1, padx=5)

        tk.Label(self.filter_frame, text="Brand:").grid(row=0, column=2, padx=5)
        self.brand_entry = tk.Entry(self.filter_frame)
        self.brand_entry.grid(row=0, column=3, padx=5)

        tk.Label(self.filter_frame, text="Size:").grid(row=1, column=0, padx=5)
        self.size_var = tk.StringVar()
        self.size_menu = ttk.Combobox(self.filter_frame, textvariable=self.size_var, values=["XS", "S", "M", "L", "XL"])
        self.size_menu.grid(row=1, column=1, padx=5)

        tk.Label(self.filter_frame, text="Material:").grid(row=1, column=2, padx=5)
        self.material_entry = tk.Entry(self.filter_frame)
        self.material_entry.grid(row=1, column=3, padx=5)

        tk.Label(self.filter_frame, text="Style:").grid(row=2, column=0, padx=5)
        self.style_var = tk.StringVar()
        self.style_menu = ttk.Combobox(self.filter_frame, textvariable=self.style_var, values=["Casual", "Formal", "Sporty"])
        self.style_menu.grid(row=2, column=1, padx=5)

        tk.Label(self.filter_frame, text="Occasion:").grid(row=2, column=2, padx=5)
        self.occasion_var = tk.StringVar()
        self.occasion_menu = ttk.Combobox(self.filter_frame, textvariable=self.occasion_var, values=["Party", "Office", "Wedding"])
        self.occasion_menu.grid(row=2, column=3, padx=5)

        tk.Label(self.filter_frame, text="Season:").grid(row=3, column=0, padx=5)
        self.season_var = tk.StringVar()
        self.season_menu = ttk.Combobox(self.filter_frame, textvariable=self.season_var, values=["Summer", "Winter", "Fall", "Spring"])
        self.season_menu.grid(row=3, column=1, padx=5)

        tk.Label(self.filter_frame, text="Pattern:").grid(row=3, column=2, padx=5)
        self.pattern_var = tk.StringVar()
        self.pattern_menu = ttk.Combobox(self.filter_frame, textvariable=self.pattern_var, values=["Plain", "Striped", "Checked"])
        self.pattern_menu.grid(row=3, column=3, padx=5)

        tk.Label(self.filter_frame, text="Price Range:").grid(row=4, column=0, padx=5)
        self.price_min = tk.Entry(self.filter_frame, width=10)
        self.price_min.grid(row=4, column=1, padx=5)
        self.price_max = tk.Entry(self.filter_frame, width=10)
        self.price_max.grid(row=4, column=2, padx=5)

        show_recommendations_button = tk.Button(self, text="Show Recommendations", command=self.show_recommendations)
        show_recommendations_button.pack(pady=10)

        self.recommendation_labels = []
        for i in range(10):
            recommendation_label = tk.Label(self)
            recommendation_label.pack(side=tk.LEFT, padx=5)
            self.recommendation_labels.append(recommendation_label)

        back_button = tk.Button(self, text="Back", command=lambda: controller.show_frame('ImageUploadPage'))
        back_button.pack(pady=10)

    def show_recommendations(self):
        features = self.controller.features
        filters = {
            'color': self.color_var.get(),
            'brand': self.brand_entry.get(),
            'size': self.size_var.get(),
            'material': self.material_entry.get(),
            'style': self.style_var.get(),
            'occasion': self.occasion_var.get(),
            'season': self.season_var.get(),
            'pattern': self.pattern_var.get(),
            'price_range': (int(self.price_min.get()), int(self.price_max.get())) if self.price_min.get() and self.price_max.get() else None
        }
        indices = recommend(features, feature_list, filters)

        for i in range(10):
            if i < len(indices[0]):
                img_path = filenames[indices[0][i]]
                img = Image.open(img_path)
                img = img.resize((200, 200))
                img_tk = ImageTk.PhotoImage(img)
                self.recommendation_labels[i].config(image=img_tk)
                self.recommendation_labels[i].image = img_tk
            else:
                self.recommendation_labels[i].config(image='')

class WishlistPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Wishlist and Favorites", font=('Helvetica', 14, 'bold'))
        label.pack(pady=10)

        self.wishlist_label = tk.Label(self, text="Your Wishlist is empty.")
        self.wishlist_label.pack(pady=10)

        back_button = tk.Button(self, text="Back", command=lambda: controller.show_frame('RecommendationPage'))
        back_button.pack(pady=10)

if __name__ == "__main__":
    app = FashionApp()
    app.mainloop()


'''







import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import os
import cv2

# Load pre-trained model and features
feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

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

class FashionRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Man & Women Fashion Recommender System")
        self.root.geometry("1200x800")

        self.frames = {}
        for F in (LandingPage, SignUpPage, LoginPage, UserProfilePage, RecommendationPage, SearchPage, CartPage, WishlistPage, AdminPanel):
            page_name = F.__name__
            frame = F(parent=self.root, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("LandingPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class LandingPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Welcome to the Fashion Recommender System", font=("Arial", 24))
        label.pack(pady=20)

        description = tk.Label(self, text="Get personalized fashion recommendations based on your style preferences.", font=("Arial", 14))
        description.pack(pady=10)

        sign_up_button = tk.Button(self, text="Sign Up", command=lambda: controller.show_frame("SignUpPage"))
        sign_up_button.pack(pady=10)

        login_button = tk.Button(self, text="Login", command=lambda: controller.show_frame("LoginPage"))
        login_button.pack(pady=10)

class SignUpPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Sign Up", font=("Arial", 24))
        label.pack(pady=20)

        self.name_entry = tk.Entry(self)
        self.name_entry.pack(pady=5)
        self.name_entry.insert(0, "Name")

        self.email_entry = tk.Entry(self)
        self.email_entry.pack(pady=5)
        self.email_entry.insert(0, "Email")

        self.password_entry = tk.Entry(self, show='*')
        self.password_entry.pack(pady=5)
        self.password_entry.insert(0, "Password")

        sign_up_button = tk.Button(self, text="Sign Up", command=self.sign_up)
        sign_up_button.pack(pady=20)

        login_label = tk.Label(self, text="Already have an account? Login", fg="blue", cursor="hand2")
        login_label.pack(pady=10)
        login_label.bind("<Button-1>", lambda e: controller.show_frame("LoginPage"))

    def sign_up(self):
        name = self.name_entry.get()
        email = self.email_entry.get()
        password = self.password_entry.get()
        # Here, you would normally add the user data to your database
        messagebox.showinfo("Success", "Account created successfully!")
        self.controller.show_frame("LoginPage")

class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Login", font=("Arial", 24))
        label.pack(pady=20)

        self.email_entry = tk.Entry(self)
        self.email_entry.pack(pady=5)
        self.email_entry.insert(0, "Email")

        self.password_entry = tk.Entry(self, show='*')
        self.password_entry.pack(pady=5)
        self.password_entry.insert(0, "Password")

        login_button = tk.Button(self, text="Login", command=self.login)
        login_button.pack(pady=20)

        sign_up_label = tk.Label(self, text="Don't have an account? Sign Up", fg="blue", cursor="hand2")
        sign_up_label.pack(pady=10)
        sign_up_label.bind("<Button-1>", lambda e: controller.show_frame("SignUpPage"))

    def login(self):
        email = self.email_entry.get()
        password = self.password_entry.get()
        # Here, you would normally check the user credentials
        messagebox.showinfo("Success", "Logged in successfully!")
        self.controller.show_frame("UserProfilePage")

class UserProfilePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="User Profile", font=("Arial", 24))
        label.pack(pady=20)

        self.name_label = tk.Label(self, text="Name: John Doe")
        self.name_label.pack(pady=5)

        self.email_label = tk.Label(self, text="Email: john.doe@example.com")
        self.email_label.pack(pady=5)

        self.profile_pic = tk.Label(self, text="Profile Picture")
        self.profile_pic.pack(pady=5)

        self.preferences_label = tk.Label(self, text="Preferences")
        self.preferences_label.pack(pady=5)

        self.past_purchases_label = tk.Label(self, text="Past Purchases")
        self.past_purchases_label.pack(pady=5)

        recommend_button = tk.Button(self, text="Get Recommendations", command=lambda: controller.show_frame("RecommendationPage"))
        recommend_button.pack(pady=20)

        search_button = tk.Button(self, text="Search for Items", command=lambda: controller.show_frame("SearchPage"))
        search_button.pack(pady=10)

class RecommendationPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Recommended Items", font=("Arial", 24))
        label.pack(pady=20)

        self.recommend_frame = tk.Frame(self)
        self.recommend_frame.pack(pady=10)

        upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
        upload_button.pack(pady=20)

        self.uploaded_image_path = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.uploaded_image_path = file_path
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            display_label = tk.Label(self, image=img)
            display_label.image = img
            display_label.pack(pady=10)
            self.recommend_fashion()

    def recommend_fashion(self):
        if self.uploaded_image_path:
            features = extract_feature(self.uploaded_image_path, model)
            indices = recommend(features, feature_list)

            for widget in self.recommend_frame.winfo_children():
                widget.destroy()

            for idx in range(1, 11):
                img_path = filenames[indices[0][idx]]
                img = Image.open(img_path)
                img = img.resize((100, 100), Image.LANCZOS)
                img = ImageTk.PhotoImage(img)
                label = tk.Label(self.recommend_frame, image=img)
                label.image = img
                label.grid(row=0, column=idx-1, padx=5)

class SearchPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Search for Items", font=("Arial", 24))
        label.pack(pady=20)

        search_bar = tk.Entry(self)
        search_bar.pack(pady=10)
        search_bar.insert(0, "Search...")

        search_button = tk.Button(self, text="Search")
        search_button.pack(pady=10)

        self.search_results_frame = tk.Frame(self)
        self.search_results_frame.pack(pady=10)

class CartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Shopping Cart", font=("Arial", 24))
        label.pack(pady=20)

        self.cart_items_frame = tk.Frame(self)
        self.cart_items_frame.pack(pady=10)

        checkout_button = tk.Button(self, text="Checkout")
        checkout_button.pack(pady=20)

class WishlistPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Wishlist", font=("Arial", 24))
        label.pack(pady=20)

        self.wishlist_frame = tk.Frame(self)
        self.wishlist_frame.pack(pady=10)

class AdminPanel(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Admin Panel", font=("Arial", 24))
        label.pack(pady=20)

        user_management_button = tk.Button(self, text="User Management")
        user_management_button.pack(pady=10)

        product_management_button = tk.Button(self, text="Product Management")
        product_management_button.pack(pady=10)

        analytics_button = tk.Button(self, text="Analytics Dashboard")
        analytics_button.pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = FashionRecommenderApp(root)
    root.mainloop()
