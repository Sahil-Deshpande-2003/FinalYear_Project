import os
from python_eulerian_video_magnification.magnifycolor import MagnifyColor
from python_eulerian_video_magnification.metadata import MetaData
from python_eulerian_video_magnification.mode import Mode
import cv2
import numpy as np
import glob


'''
    Magnification video
'''
def rename_processed_video(mag_path, original_vidname):
    """
    Rename the processed video to remove the timestamp added by MetaData.
    """
    pattern = os.path.join(mag_path, original_vidname.split('.')[0] + "_*.avi")  # Find matching files
    files = glob.glob(pattern)

    for file_path in files:
        base_name = os.path.basename(file_path)
        new_name = original_vidname  # Rename back to original name
        new_path = os.path.join(mag_path, new_name)
        
        os.rename(file_path, new_path)
        print(f"Renamed: {file_path} -> {new_path}")
        
def generate_mag_video(vid_path, mag_path):
    if not os.path.exists(mag_path):
        os.makedirs(mag_path)

    vidlist = sorted(os.listdir(vid_path))  # Sort video files

    for vidname in vidlist:
        print(f"Processing: {vidname} ...", end='', flush=True)
        vidpath = os.path.join(vid_path, vidname)  # Input video path

        #Fix: Using correct MetaData arguments
        metadata = MetaData(
            file_name=vidpath,  # Input video
            output_folder=mag_path,  # Directory where output will be stored
            mode=Mode.COLOR,  # Magnification mode
            suffix="",  # No extra suffix
            low=0.833, high=2,  # Frequency range
            levels=1,  # Number of pyramid levels
            amplification=10  # Magnification factor
        )

        MagnifyColor(metadata).do_magnify()

        #Rename file to remove timestamp
        rename_processed_video(mag_path, vidname)

        print(" Done!")


if __name__=="__main__":

    video_dir_name = "original_sequences/youtube/c23/"
    #video_dir_name = "manipulated_sequences/Deepfakes/c23/"

    data_root_dir = "/kaggle/working/01_0_align_original_video/"
    datadir = data_root_dir + video_dir_name

    new_vid_root_dir = "/kaggle/working/02_mag_video/"
    newviddir = new_vid_root_dir + video_dir_name
    if not os.path.exists(newviddir):
        os.makedirs(newviddir)
        
    generate_mag_video(datadir, newviddir)