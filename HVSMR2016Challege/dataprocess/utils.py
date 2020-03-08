from __future__ import print_function, division
import os
import numpy as np
import pandas as pd


def calcu_dice(Y_pred, Y_gt, K=255):
    """
    calculate two input dice value
    :param Y_pred:
    :param Y_gt:
    :param K:
    :return:
    """
    intersection = 2 * np.sum(Y_pred[Y_gt == K])
    denominator = np.sum(Y_pred) + np.sum(Y_gt) + 1e-5
    loss = (intersection / denominator)
    return loss


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


def check_trained_data(csv_path, ratio=0.001):
    csvdata = pd.read_csv(csv_path)
    maskdata = csvdata.iloc[:, 1].values
    count = 0
    for num in range(len(maskdata)):
        mask = np.load(maskdata[num])
        labelpixel_count = (mask != 0).sum()
        backgroundpixel_count = (mask == 0).sum()
        pixel_ratio = labelpixel_count / backgroundpixel_count
        if backgroundpixel_count != 0 and pixel_ratio > ratio:
            count = count + 1
    print(count / len(maskdata))


def save_file2csv(file_dir, file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    image = "Image"
    mask = "Mask"
    file_image_dir = file_dir + "/" + image
    file_mask_dir = file_dir + "/" + mask
    file_paths = file_name_path(file_image_dir, dir=False, file=True)
    out.writelines("Image,Mask" + "\n")
    for index in range(len(file_paths)):
        out_file_image_path = file_image_dir + "/" + file_paths[index]
        out_file_mask_path = file_mask_dir + "/" + file_paths[index]
        out.writelines(out_file_image_path + "," + out_file_mask_path + "\n")


if __name__ == "__main__":
    save_file2csv("E:\MedicalData\HVSMR2016\\traindata", "train1mm.csv")
