"""
Authors : Anil Kunchala, Melanie Bouroche,Bianca Schoen-Phelan
Email : d20125529@mytudublin.ie
"""
import os
import numpy as np
import torch
import torchvision

def create_dir_if_not_exist(dir_to_check):
    os.makedirs(dir_to_check,exist_ok=True)

def unzip_file(zip_file, dir_to_unzip="tmp_zip"):
    print(F"unzipping {zip_file} to {dir_to_unzip}")
    cmd_to_run = F"unzip -j -qq {zip_file}" # -j used to not to create an dir https://unix.stackexchange.com/questions/72838/unzip-file-contents-but-without-creating-archive-folder

    create_dir_if_not_exist(dir_to_unzip)
    #change the current dir to the dir_to_unzip
    ROOT_DIR = os.getcwd()
    dir_to_unzip = os.path.join(ROOT_DIR, dir_to_unzip)
    os.chdir(dir_to_unzip)
    # remove all the files before unzipping
    os.system(F"rm *.jpg *.avi *.mp4")
    
    os.system(cmd_to_run)
    os.chdir(ROOT_DIR)
    return dir_to_unzip

def convert_video_to_images(video_file,dir_to_store="tmp"):
    print(F"Converting {video_file} to images and storing in {dir_to_store}")
    cmd_to_run = F"ffmpeg -hide_banner -loglevel error -i {video_file} 'img_%05d.jpg' -start_number 0" # get the frames using default frame rate

    #change the current dir to dir_to_store
    # os.system(F"cd {dir_to_store}")
    # https://stackoverflow.com/questions/35277128/why-does-not-os-systemcd-mydir-work-and-we-have-to-use-os-chdirmydir-ins
    ROOT_DIR = os.getcwd()
    dir_to_store = os.path.join(ROOT_DIR,dir_to_store)
    os.chdir(dir_to_store)

    # remove all the files in this dir
    os.system (F"rm *.jpg")
    # print(cmd_to_run)
    # run the command to generate the images
    os.system(cmd_to_run)

    os.chdir(ROOT_DIR)

    return dir_to_store   

