import os
import cv2
import numpy as np
import contextlib
from preprocessing.Mesonet.classifiers import MesoInception4


def process_meso_data(real_video_dir, fake_video_dir, fake_mit_dir, real_mit_dir, model_weights_path, save_path):
    save_path = "./EX_STORE/Beauty_app/df_ytb_c23/"
    classifier = MesoInception4()
    classifier.load(model_weights_path)
    
    real_vid_list = sorted(os.listdir(real_video_dir))
    fake_vid_list = sorted(os.listdir(fake_video_dir))
    
    video_dict = {x: os.path.join(real_video_dir, x) for x in real_vid_list}
    video_dict.update({x: os.path.join(fake_video_dir, x) for x in fake_vid_list})
    
    data_name_list = real_vid_list + fake_vid_list
    
    fake_mit_dict = {x: os.path.join(fake_mit_dir, x, f'{x}.npy') for x in os.listdir(fake_mit_dir)}
    real_mit_dict = {x: os.path.join(real_mit_dir, x, f'{x}.npy') for x in os.listdir(real_mit_dir)}
    
    mit_dict = {**fake_mit_dict, **real_mit_dict}
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    data_Meso_set, data_mit_set, data_y_set, data_name_set = [], [], [], []
    
    for vid_name in data_name_list:
        print(f"Processing video: {vid_name}", end='', flush=True)
        
        if vid_name + ".avi" in mit_dict:
            print(" y")
            
            vid_path = os.path.join(video_dict[vid_name])
            img_list = sorted([img for img in os.listdir(vid_path) if "_face" not in img])
            if not img_list:
                continue
            
            data_Meso = np.ones(300) * 0.5
            try:
                for img_name in img_list:
                    img_index = int(img_name[:-4])
                    if img_index >= 300:
                        continue
                    
                    img_path = os.path.join(vid_path, img_name)
                    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                        img = cv2.resize(cv2.imread(img_path), (256, 256))
                        pred = classifier.predict(np.array([img]))
                        data_Meso[img_index] = pred
            except Exception:
                continue
            
            data_Meso_set.append(data_Meso)
            data_mit_set.append(np.load(mit_dict[vid_name + ".avi"]))
            data_y_set.append(1 if "_" in vid_name else 0)
            data_name_set.append(vid_name)
            print(data_y_set[-1])
    
    print("SAVING .... ")
    np.save(os.path.join(save_path, "Meso.npy"), data_Meso_set)
    np.save(os.path.join(save_path, "mit.npy"), data_mit_set)
    np.save(os.path.join(save_path, "y.npy"), data_y_set)
    np.save(os.path.join(save_path, "name.npy"), data_name_set)
    
    print("Processing complete!")
