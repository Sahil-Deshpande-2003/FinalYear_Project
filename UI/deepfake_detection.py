# from preprocessing.00_align_face_with_mtcnn import generate_align_face model names cant start with numbers

from preprocessing.align_face_with_mtcnn import generate_align_face
from preprocessing.resize_frame import resize_frame
from preprocessing.align_video import generate_align_video
from preprocessing.gen_motion_mag_video import generate_mag_video
from preprocessing.gen_map import generate_mmst_map
from preprocessing.gen_training_data_modified import process_meso_data
from preprocessing.Mesonet.classifiers import MesoInception4
from  preprocessing.paths import UPLOAD_FOLDER
import os
def preprocess_video(video_path,video_name):
  
    face_align_path = generate_align_face(video_path,video_name)

    resize_path = resize_frame(face_align_path,video_name)

    align_path = generate_align_video(resize_path,video_name)

    mag_path = generate_mag_video(align_path,video_name)

    map_path = generate_mmst_map(mag_path,video_name)

    print("\n")

    print("LIFE AFTER MAP!!!\n")

    classifier = MesoInception4()

    print("LIFE AFTER classifier!!!\n")

    Meso_data = process_meso_data(video_name,classifier)
    

    print("Step-1 done")

    return 0


