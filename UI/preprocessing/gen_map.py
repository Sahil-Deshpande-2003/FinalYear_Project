
import os
import cv2
import preprocessing.util_mit as util
import numpy as np

newviddir = "./EX_STORE/Beauty_app/dfdc/stmap/"

'''
    Use magnified video to produce mmst-map
'''
def generate_mmst_map(mag_path, map_path):

    map_path = newviddir

    if not os.path.exists(map_path):
        os.makedirs(map_path)

    ROI_h, ROI_w = 5, 5

    vidlist = sorted([f for f in os.listdir(mag_path) if f.endswith('.avi')])
    for vidname in vidlist:
        print("{} - {} ... ".format(mag_path, vidname), end='', flush=True)

        vidpath = mag_path + vidname
        vid = cv2.VideoCapture(vidpath)
        full_st_map = np.zeros((300, 25, 3))
        idx = 0
        while idx < 300:
            success, frame = vid.read()
            if not success:
                break
            # print("  %3d/300 ... " % idx, end='', flush=True)
            frame_seg = util.get_frame_seg(frame, ROI_h, ROI_w)
            full_st_map[idx] = frame_seg
            idx += 1
            # print("Done")
        
        vidname_clean = vidname  # Removes ".avi"
        save_vid_path = os.path.join(map_path, vidname_clean)
        os.makedirs(save_vid_path, exist_ok=True)
        np.save(os.path.join(save_vid_path, vidname_clean + ".npy"), full_st_map)


