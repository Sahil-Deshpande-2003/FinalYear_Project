import os
import json
import dlib
from facenet_pytorch import MTCNN
import numpy as np
import preprocessing.util_img as util
import cv2
import torch
from preprocessing.paths import UPLOAD_FOLDER

print("Preparing dlib ... ", end='', flush=True)
detector = dlib.get_frontal_face_detector()
predictor_path = r'D:\Btech_Project\UI\preprocessing\shape_predictor_81_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
print("Done")
print("Preparing MTCNN ... ", end='', flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(thresholds=[0.3, 0.3, 0.3], margin=20, keep_all=True, post_process=False, select_largest=False, device=device)
print("Done")


meta_dir = './EX_STORE/Beauty_app/00_Face_data/' 

if not os.path.exists(meta_dir):
    os.makedirs(meta_dir)

'''
    Generate aligned face and remove background
'''
def generate_align_face(video_path, video_name):

    vidpath = video_path

    print(f"Uploaded video path = {vidpath}")

    print(f"Uploaded video name = {video_name}")

    save_vid_path = meta_dir + video_name + '/'

    print(f"save_vid_path = {save_vid_path}")

    if os.path.exists(save_vid_path):
        pass
    else:
        os.mkdir(save_vid_path)
    
    try:
        align_face = util.preprocess_video(video_name, detector, predictor, mtcnn, vidpath, save_vid_path)
    except:
        print("Error caused !!")
    
    return save_vid_path


