import os
import cv2
import numpy as np

# datadir = "/kaggle/working/02_mag_video/" => MADE THIS CHANGE!!

datadir = "/kaggle/working/motion_mag_video_f2f_1000/manipulated_sequences/Face2Face/c23/"

newviddir = "/kaggle/working/dfdc_f2f_1000/stmap/"
'''
    Use magnified video to produce mmst-map
'''
def generate_mmst_map(mag_path, map_path):
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    # print(f"mag_path = {mag_path}")
    # print(f"map_path = {map_path}")
    ROI_h, ROI_w = 5, 5

    vidlist = os.listdir(mag_path)
    vidlist.sort()
    for vidname in vidlist:
        vidpath = os.path.join(mag_path, vidname)
        if os.path.isdir(vidpath):  # If it's a folder, look for .avi files inside
            avi_files = [f for f in os.listdir(vidpath) if f.endswith('.avi')]
            if avi_files:
                vidpath = os.path.join(vidpath, avi_files[0])  # Use the first .avi file  # DANGER!!!!!!
            else:
                print(f"No .avi files found in directory {vidpath}")
                continue

        if not os.path.exists(vidpath):
            print(f"Error: File does not exist at {vidpath}")
            continue

        vid = cv2.VideoCapture(vidpath)
        if not vid.isOpened():
            print(f"Error: Cannot open video file {vidpath}")
            continue

        # print(f"Processing video - {vidpath}")
        full_st_map = np.zeros((300, 25, 3))
        idx = 0
        while idx < 300:
            success, frame = vid.read()
            if not success:
                print(f"End of video or read error at frame {idx}.")
                break
            frame_seg = get_frame_seg(frame, ROI_h, ROI_w)
            full_st_map[idx] = frame_seg
            idx += 1

        save_vid_path = os.path.join(map_path, vidname + ".npy")
        np.save(save_vid_path, full_st_map)
        # print('full_st_map = ' + str(full_st_map))
        # print(f"Saved ST map for {vidname}")


if __name__=="__main__":
    generate_mmst_map(datadir, newviddir)