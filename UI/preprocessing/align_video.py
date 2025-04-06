import os
import cv2
import numpy as np

new_vid_root_dir = "./EX_STORE/Beauty_app/01_0_align_original_video/"

'''
    Generate video with aligned face. These videos will be used in motion magnificaiton.
'''
def generate_align_video(resize_path, video_name):

    if not os.path.exists(new_vid_root_dir):
        os.makedirs(new_vid_root_dir)


    vidpath = resize_path

    save_vid_path = new_vid_root_dir + video_name + '.avi'
        
    os.system("ffmpeg -i {}%04d.jpg {}".format(vidpath, save_vid_path))
    print("Done")

    return save_vid_path
