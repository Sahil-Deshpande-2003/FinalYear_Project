# !sudo apt install cmake
# !pip install face_recognition
mesonet_repo_path = "/kaggle/input/mesonetrewa"
import sys
sys.path.append(mesonet_repo_path) # sys.path is a list of dirs where Python searches for the modules, and hence this dir is totally different from curr workingdir /kaggle/working, hence we need to append it separately
import numpy as np
from classifiers import *
from pipeline import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import cv2

# Ensure TensorFlow uses only CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Load the MesoInception4 model with pretrained weights
classifier = MesoInception4()
classifier.load('/kaggle/input/mesonetrewa/weights/MesoInception_F2F.h5')

# Directory containing deepfake videos
fake_video_dir = "/kaggle/working/resize_frames_f2f_1000/manipulated_sequences/Face2Face/c23"
fake_vid_list = os.listdir(fake_video_dir)
fake_vid_list.sort()
fake_vid_dict = {x.replace('.avi', ''): os.path.join(fake_video_dir, x) for x in fake_vid_list}

# Save directory
save_path = "model_save_path/"
if not os.path.exists(save_path):
    os.mkdir(save_path)
    print(f"Created save directory: {save_path}")

# Data storage
data_Meso_set = []
data_name_set = []

for vid_name in fake_vid_dict:
    vid_path = fake_vid_dict[vid_name] + '/'
    img_list = os.listdir(vid_path)
    img_list = [img for img in img_list if '_face' not in img]  # Ignore face-segmented images
    img_list.sort()
    
    if not img_list:
        print(f"No images found in {vid_path}. Skipping video.")
        continue
    
    data_Meso = np.ones(300) * 0.5  # Initialize array with neutral values
    try:
        for img_name in img_list:
            frame_idx = int(img_name[:-4])  # Extract frame number
            if frame_idx >= 300:
                continue  # Only consider first 300 frames
            
            img_path = os.path.join(vid_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_name}. Skipping.")
                continue
            
            img = cv2.resize(img, (256, 256))
            pred = classifier.predict(np.array([img]))
            data_Meso[frame_idx] = pred  # Store prediction at correct index
    except Exception as e:
        print(f"Error processing video {vid_name}: {e}")
        continue
    
    data_Meso_set.append(data_Meso)
    data_name_set.append(vid_name)

# Save only Meso.npy
np.save(os.path.join(save_path, "Meso_f2f.npy"), data_Meso_set)
np.save(os.path.join(save_path, "name_f2f.npy"), data_name_set)
print("Meso_f2f.npy and name_f2f.npy saved successfully for f2f videos.")
