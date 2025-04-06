import os
import cv2
import numpy as np
import shutil
resize_root_dir = "./EX_STORE/Beauty_app/01_1_resize_original_frame/"

'''
    Resize the aligned face so they can be formed into a video, which will
    be used into motion magnification.
'''
def resize_frame(face_align_path, video_name):

    if not os.path.exists(resize_root_dir):
        os.makedirs(resize_root_dir)

    vidpath = face_align_path

    print(f'RESIZE FRAME => vidpath = {vidpath}')

    save_vid_path = resize_root_dir + video_name + '/'
    print(f'RESIZE FRAME => save_vid_path = {save_vid_path}')
    os.mkdir(save_vid_path)

    imglist = os.listdir(vidpath)
    imglist.sort()
    f_h, f_w = None, None
    # cnt = 0
    for imgname in imglist:
        # cnt+=1
        # if (cnt == 20):
            # print(f"cnt = {cnt} and breaking...")
            # break
        if imgname.endswith('.npy'):
            continue
            
        if imgname.endswith('_face.jpg'):
            continue

        imgpath = vidpath + imgname

        img = cv2.imread(imgpath)
        if f_h is None:
            f_h, f_w = img.shape[:2]
            f_h = ((int(f_h/10)+1)*10)
            f_w = ((int(f_w/10)+1)*10)

        img = cv2.resize(img, (f_w, f_h))

        save_img_path = save_vid_path + imgname
        cv2.imwrite(save_img_path, img)

        idx = None
        imglist = os.listdir(save_vid_path)
        imglist.sort()
        imglist.append("0300.jpg")
        for imgname in imglist:
            if int(imgname[:4]) >= 300:
                break
            if idx is None and int(imgname[:4])!=0:
                for i in range(0, int(imgname[:4])):
                    ori_img = save_vid_path + imgname
                    new_img = save_vid_path + ("%04d.jpg"%i)
                    shutil.copy(ori_img, new_img)
            if idx is not None and idx < int(imgname[:4]):
                for i in range(idx, int(imgname[:4])):
                    ori_img = save_vid_path + ("%04d.jpg"%(idx-1))
                    new_img = save_vid_path + ("%04d.jpg"%i)
                    shutil.copy(ori_img, new_img)
            idx = int(imgname[:4]) + 1
        print("Done")

    
    return save_vid_path
        
