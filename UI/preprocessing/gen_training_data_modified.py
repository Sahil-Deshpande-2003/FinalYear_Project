import os
import cv2
import numpy as np
import contextlib

base_path = r'EX_STORE\Beauty_app\01_1_resize_original_frame'

def process_meso_data(video_name,classifier):
    """
    Processes images in the given video path using the MesoInception4 classifier.

    Parameters:
    - video_name: str, name of the video
    - vid_path: str, path to the video directory
    - img_list: list of image filenames
    - classifier: MesoInception4 model instance
    - data_Meso: dict, stores predictions indexed by image number
    """
    data_Meso = np.ones(300) * 0.5
    
    img_list_tmp = os.path.join(base_path, video_name)  # Construct full video path

    img_list = os.listdir(img_list_tmp)

    print(f"img_list = {img_list}\n")
    
    for img_name in img_list:
        try:
            if "_face" in img_name:
                continue

            # print(f'img_name[:-4] = {img_name[:-4]}\n')

            img_index = int(img_name[:-4])  # Extracting image index from filename
            if img_index >= 300:
                continue

            img_path_tmp = os.path.join(base_path, video_name)

            img_path = os.path.join(img_path_tmp,img_name)

            # print("All Good before with!!")

            # print(f'path = {os.path.abspath(img_path)}')

            # with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            with open(os.devnull, 'w', encoding="utf-8") as f, contextlib.redirect_stdout(f):
                img = cv2.resize(cv2.imread(img_path), (256, 256))
                pred = classifier.predict(np.array([img]))
                data_Meso[img_index] = pred


        except Exception as e:
            print(f"Error processing {img_name}: {e}")  # Logging errors instead of silent failure

    np.save("Meso.npy", data_Meso)
    print("Predictions saved to Meso.npy")

    return data_Meso