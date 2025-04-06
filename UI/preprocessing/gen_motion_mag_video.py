
import os
import glob
from python_eulerian_video_magnification.magnifycolor import MagnifyColor
from python_eulerian_video_magnification.metadata import MetaData
from python_eulerian_video_magnification.mode import Mode
import cv2
import numpy as np


mag_path = "./EX_STORE/Beauty_app/02_mag_video/"


'''
    Magnification video
'''

def rename_processed_video(mag_path, original_vidname):
    print("INSIDE RENAME!!!!!!!")

    mag_path = os.path.abspath(mag_path)

    # Extract base filename without extension
    base_name = os.path.splitext(original_vidname)[0]  # Removes .mp4 from original_vidname
    
    # Corrected pattern to include '.mp4' before '_evm'
    pattern = os.path.join(mag_path, f"{base_name}.mp4__evm_*.avi")

    print(f"Searching for files matching: {pattern}")

    # List all files in the directory
    print(f"Files in {mag_path}: {os.listdir(mag_path)}")



    files = glob.glob(pattern)
    print(f"Matched files: {files}")

    if not files:
        print("No files matched the pattern. Check the filename format and ensure the output files exist.")
        return

    for file_path in files:
        new_name = f"{base_name}.avi"  # Rename back to original name with .avi
        new_path = os.path.join(mag_path, new_name)

        try:
            os.rename(file_path, new_path)
            print(f"Renamed: {file_path} -> {new_path}")
        except Exception as e:
            print(f"Error renaming {file_path}: {e}")

def generate_mag_video(vid_path, vidname):
    if not os.path.exists(mag_path):
        os.makedirs(mag_path)

    metadata = MetaData(
        file_name=vid_path,
        output_folder=mag_path,
        mode=Mode.COLOR,
        suffix="",
        low=0.833, high=2,
        levels=1,
        amplification=10
    )

    MagnifyColor(metadata).do_magnify()

    # Rename file to .avi
    rename_processed_video(mag_path, vidname)

    return mag_path

    print("Done!")
