#01_2_align_video.py
import os
import cv2
import numpy as np

'''
    Generate video with aligned face. These videos will be used in motion magnificaiton.
'''
def generate_align_video(video_frame_path, video_store_path):
    if not os.path.exists(video_store_path):
        os.makedirs(video_store_path)

    vidlist = os.listdir(video_frame_path)
    vidlist.sort()
    for vidname in vidlist:
        print("{} - {} ... ".format(video_frame_path, vidname), end='', flush=True)
        vidpath = video_frame_path + vidname +'/'

        save_vid_path = video_store_path + vidname + '.avi'
            
        os.system("ffmpeg -i {}%04d.jpg {}".format(vidpath, save_vid_path))
        print("Done")


if __name__=="__main__":

    #video_dir_name = "original_sequences/youtube/c23/"
    video_dir_name = "manipulated_sequences/Face2Face/c23/"

    data_root_dir = "/kaggle/working/resize_frames_f2f_1000/"
    datadir = data_root_dir + video_dir_name

    new_vid_root_dir = "/kaggle/working/align_video_f2f_1000/"
    newviddir = new_vid_root_dir + video_dir_name
    if not os.path.exists(newviddir):
        os.makedirs(newviddir)

    generate_align_video(datadir, newviddir)