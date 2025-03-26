# !pip install https://github.com/vgoehler/PyEVM/archive/master.zip # installs the in-development version of Eulerian Video Magnification 
import os
from python_eulerian_video_magnification.magnifycolor import MagnifyColor
from python_eulerian_video_magnification.metadata import MetaData
from python_eulerian_video_magnification.mode import Mode
import cv2
import numpy as np

# I have added this function temporarily
def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

'''
    Magnification video
'''
def generate_mag_video(vid_path, mag_path):
    if not os.path.exists(mag_path):
        os.makedirs(mag_path)
    # print('vid_path = ' + str(vid_path))
    # print('mag_path = ' + str(mag_path))
    vidlist = os.listdir(vid_path)
    vidlist.sort()
    # print('vidlist = ' + str(vidlist))
    for vidname in vidlist:
        # print("{} - {} ... ".format(original_video_dir_name, vidname), end='', flush=True)
        # print('\n')
        vidpath = vid_path + vidname
        # print('vidpath = ' + str(vid_path))

        save_vid_path = mag_path + vidname

        # print('save_vid_path = ' + str(save_vid_path))

        ensure_dir_exists(save_vid_path)
        # print('before this line!')
        MagnifyColor(MetaData(file_name=vidpath, low=0.833, high=2, levels=1,
                    amplification=10, output_folder=save_vid_path, mode=Mode.COLOR, suffix='color')).do_magnify() 
        # print('after this line!')
        # MagnifyColor(MetaData(file_name=vidpath, low=0.833, high=2, levels=1,
        #             amplification=10, target_path=save_vid_path, output_folder='', mode=Mode.COLOR, suffix='color')).do_magnify() 
        # print("Done")

if __name__=="__main__":

    # data_root_dir = "/kaggle/working/01_2_align_video/"
    
    data_root_dir =  "/kaggle/working/align_video_f2f_1000/"
    original_video_dir_name = "manipulated_sequences/Face2Face/c23/"
    
    datadir = data_root_dir + original_video_dir_name

    new_vid_root_dir = "/kaggle/working/motion_mag_video_f2f_1000/"
   
    newviddir = new_vid_root_dir + original_video_dir_name
    if not os.path.exists(newviddir):
        os.makedirs(newviddir)
        
    generate_mag_video(datadir, newviddir)