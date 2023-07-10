"""
Authors : Anil Kunchala, Melanie Bouroche,Bianca Schoen-Phelan
Email : d20125529@mytudublin.ie

"""

import os
import shutil
import cv2

import utils
import person_detector


#TODO - add walking towards and walking away at later stage
NTU_RGB_SELECTED_ACTIVITIES = {
    "A008" : "Sit_down",
    "A009" : "Stand_up",
    "A023" : "Hand_waving",
    "A028" : "Phone_call",
    "A029" : "Play_with_phone_or_tablet",
    "A032" : "Taking_a_selfie",
    "A099" : "Run_on_the_spot"
}


def get_ntu_rgb_tracklets(src_dir,des_dir) :
    """ process ntu rgb dataset zip files and extract both frames and tracklets

    Args :
        src_dir: root dir of dataset with zip files
        des_dir: des dir to store frames and tracklets
    
    Return :
        None
    """
    
    utils.create_dir_if_not_exist(des_dir)
    dir_to_store_frames = os.path.join(des_dir,"ntu_rgb_frames")
    utils.create_dir_if_not_exist(dir_to_store_frames)
    dir_to_store_tracklets = os.path.join(des_dir,"ntu_rgb_tracklets")
    utils.create_dir_if_not_exist(dir_to_store_tracklets)

    all_zip_files = os.listdir(src_dir)
    for f_name in all_zip_files :
        f_name = os.path.join(src_dir,f_name)
        if os.path.isfile(f_name) and f_name.split(".")[-1] == 'zip' : # check if its a zip file
            # video_files_dir = utils.unzip_file(f_name)
            video_files_dir = "/home/akunchala/Documents/PhDStuff/action_tracklet_parser/tmp_zip"
            for _video_f in os.listdir(video_files_dir) :
                video_f = os.path.join(video_files_dir,_video_f)
                act_idx = F"A{video_f.split('A')[1].split('_')[0]}"

                if act_idx in NTU_RGB_SELECTED_ACTIVITIES.keys() :
                    utils.create_dir_if_not_exist(os.path.join(dir_to_store_frames,act_idx))
                    utils.create_dir_if_not_exist(os.path.join(dir_to_store_tracklets,act_idx))
                    print(F"Processing {video_f}")
                    frames_dir = utils.convert_video_to_images(video_f)
                    bboxes_detections = person_detector.get_person_bboxes_from_dir(frames_dir)
                    out_frame_dir = os.path.join(dir_to_store_frames,act_idx,F"{os.path.basename(video_f).split('.')[0]}")
                    out_tracklet_dir = os.path.join(dir_to_store_tracklets,act_idx,F"{os.path.basename(video_f).split('.')[0]}")
                    utils.create_dir_if_not_exist(out_frame_dir)
                    utils.create_dir_if_not_exist(out_tracklet_dir)
                    f_name_index = 0
                    for f_idx in range(0, len(os.listdir(frames_dir))) :
                        # for some weird reason ffmpeg generating idx from 1 (reason for +1)
                        frame_f = os.path.join(frames_dir,F"img_{f_idx+1:05d}.jpg")

                        #copy it into frames_dir
                        frames_des = os.path.join(out_frame_dir,F"img_{f_name_index:05d}.jpg")
                        shutil.copy2(frame_f,frames_des)
                        if bboxes_detections.get(F"img_{f_idx+1:05d}", None) == None :
                            continue # ignore the missing detections 
                        # extract bbox and save it in tracklet dir
                        bbox = bboxes_detections[F"img_{f_idx+1:05d}"]
                        img = cv2.imread(frame_f)
                        crop_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                        out_f_name = os.path.join(out_tracklet_dir,F"img_{f_name_index:05d}.jpg")
                        cv2.imwrite(out_f_name,crop_img)
                        f_name_index = f_name_index + 1 # used to ignore missing detections
                    # return
                    
def check_single_video(video_file_name) :
    frames_dir = utils.convert_video_to_images(video_file_name)
    bbox_detections = person_detector.get_person_bboxes_from_dir(frames_dir)

    out_tracklet_dir = "ntu_tracklets_test"
    utils.create_dir_if_not_exist(out_tracklet_dir)
    
    f_name_index = 0

    for f_idx in range(0, len(os.listdir(frames_dir))) :
        frame_i = os.path.join(frames_dir,F"img_{f_idx+1:05d}.jpg")
        # print(frame_i)

        bbox = bbox_detections[F"img_{f_idx+1:05d}"]
        img = cv2.imread(frame_i)
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        frames_des = os.path.join(out_tracklet_dir,F"img_{f_name_index:05d}.jpg")
        cv2.imwrite(frames_des,crop_img)
        f_name_index += 1




if __name__ == "__main__" :
    SRC_DIR = "NTU_RGB"
    DIR_TO_STORE = "NTU_TRACKLET_DATA"
    get_ntu_rgb_tracklets(SRC_DIR,DIR_TO_STORE)

    # test_ntu_video = "NTU_RGB/nturgb+d_rgb/S001C001P001R001A009_rgb.avi"
    # check_single_video(test_ntu_video)
