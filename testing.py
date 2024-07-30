import os
import pickle

base_dir = r"C:\Users\HP\Desktop\fashion[1]\Fashion_Recommander_System-main\Dataset"
filenames_path = r"C:\Users\HP\Desktop\fashion[1]\Fashion_Recommander_System-main\filenames.pkl"

# Check if base_dir exists
if not os.path.exists(base_dir):
    print(f"Error: The base directory '{base_dir}' does not exist.")
else:
    filenames = []
    try:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                # Append relative paths
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                filenames.append(relative_path)
    except Exception as e:
        print(f"Error walking through the directory: {e}")

    # Save filenames to pickle
    try:
        with open(filenames_path, 'wb') as f:
            pickle.dump(filenames, f)
        print("Updated filenames.pkl with the current directory state.")
    except Exception as e:
        print(f"Error saving filenames: {e}")
