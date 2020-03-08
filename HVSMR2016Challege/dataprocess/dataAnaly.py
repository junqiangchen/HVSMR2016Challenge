from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
from dataprocess.utils import file_name_path
from dataprocess.data3dprepare import resize_image_itk

heart_path = "E:\MedicalData\HVSMR2016\cropedHeart\Test dataset"
image_name = "Training dataset"
mask_name = "Ground truth"


def getImageSizeandSpacing():
    """
    get image and spacing
    :return:
    """
    mutilorgan_pathtmp = heart_path + "/" + image_name
    file_path_list = file_name_path(mutilorgan_pathtmp, False, True)
    for subsetindex in range(len(file_path_list)):
        mutilorgan_subset_path = mutilorgan_pathtmp + "/" + str(file_path_list[subsetindex])
        src = sitk.ReadImage(mutilorgan_subset_path, sitk.sitkInt16)
        imageSize = src.GetSize()
        imageSpacing = src.GetSpacing()
        print("image size,image spcing:", (imageSize, imageSpacing))


def getresizeImageSize():
    """
       get image and spacing
       :return:
       """
    mutilorgan_pathtmp = heart_path
    file_path_list = file_name_path(mutilorgan_pathtmp, False, True)
    newSpacing = (1.0, 1.0, 1.0)
    for subsetindex in range(len(file_path_list)):
        mutilorgan_subset_path = mutilorgan_pathtmp + "/" + str(file_path_list[subsetindex])
        src = sitk.ReadImage(mutilorgan_subset_path, sitk.sitkInt16)
        _, src = resize_image_itk(src, newSpacing, src.GetSpacing(), sitk.sitkLinear)
        imageSize = src.GetSize()
        imageSpacing = src.GetSpacing()
        print("image size,image spcing:", (imageSize, imageSpacing))


def getMaskLabelMaxValue():
    """
    get max mask value
    kits mask have three value:0,1,2(0 is backgroud ,1 is heart,2 is myocardium)
    :return:
    """
    mutilorgan_pathtmp = heart_path + "/" + mask_name
    file_path_list = file_name_path(mutilorgan_pathtmp, False, True)
    for subsetindex in range(len(file_path_list)):
        mask_path = mutilorgan_pathtmp + "/" + str(file_path_list[subsetindex])
        seg = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        segimg = sitk.GetArrayFromImage(seg)
        seg_maskimage = segimg.copy()
        max_value = np.max(seg_maskimage)
        print("max_mask_value:", max_value)


if __name__ == "__main__":
    # getMaskLabelMaxValue()
    # getImageSizeandSpacing()
    getresizeImageSize()
