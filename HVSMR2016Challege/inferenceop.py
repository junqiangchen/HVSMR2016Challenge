from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_mutildepth_vnet3d_multilabel import mutildepthVnet3dModuleMultiLabel
from dataprocess.utils import file_name_path
from dataprocess.data3dprepare import normalize, resize_image_itk
import numpy as np
import SimpleITK as sitk


def inference1mm():
    """
    Vnet network segmentation kidney fine segmatation
    :return:
    """
    channel = 1
    numclass = 3
    newSpacing = (1.0, 1.0, 1.0)
    # step1 init vnet model
    imagex = 256
    imagey = 256
    imagez = 208
    Vnet3d = mutildepthVnet3dModuleMultiLabel(imagex, imagey, imagez, channels=channel, numclass=numclass,
                                    costname=("categorical_dice",), inference=True,
                                    model_path="log\mutildepthVnet\categorical_dice\model\Vnet3d.pd")
    heart_path = "E:\MedicalData\HVSMR2016\cropedHeart\Test dataset"
    out_path = "E:\MedicalData\HVSMR2016\cropedHeart\Test_predict"
    # step2 get all test image path
    path_list = file_name_path(heart_path, dir=False, file=True)
    # step3 get test image(4 model) and mask
    for subsetindex in range(len(path_list)):
        # step4 load test image as ndarray
        subset_path = heart_path + "/" + str(path_list[subsetindex])
        heart_src = sitk.ReadImage(subset_path, sitk.sitkInt16)
        srcSpacing = heart_src.GetSpacing()
        srcOrigin = heart_src.GetOrigin()
        srcSize = heart_src.GetSize()
        srcDirection = heart_src.GetDirection()
        zspacing, xpacing, yspacing = srcSpacing[2], srcSpacing[0], srcSpacing[1]
        _, heart_src = resize_image_itk(heart_src, newSpacing=newSpacing,
                                        originSpcaing=(xpacing, yspacing, zspacing),
                                        resamplemethod=sitk.sitkLinear)
        heart_array = sitk.GetArrayFromImage(heart_src)
        # step5 mormazalation test image
        heart_array = normalize(heart_array)
        depthz, heighty, widthx = np.shape(heart_array)[0], np.shape(heart_array)[1], np.shape(heart_array)[2]
        vnetinputarray = np.zeros((imagez, imagey, imagex, channel), np.float)
        vnetinputarray[0:depthz, 0:heighty, 0:widthx, 0] = heart_array
        # step6 predict test image
        Vnet3d_array = Vnet3d.prediction(vnetinputarray)
        ys_pd_array = Vnet3d_array[0:depthz, 0:heighty, 0:widthx]
        ys_pd_sitk = sitk.GetImageFromArray(ys_pd_array)
        ys_pd_sitk.SetSpacing(heart_src.GetSpacing())
        ys_pd_sitk.SetOrigin(heart_src.GetOrigin())
        ys_pd_sitk.SetDirection(heart_src.GetDirection())
        # step7.1 resample output to origin size
        _, ys_pd_sitk = resize_image_itk(ys_pd_sitk, newSpacing=(xpacing, yspacing, zspacing),
                                         originSpcaing=newSpacing, resamplemethod=sitk.sitkNearestNeighbor, flag=False)
        # 7.2 make sure output size is same as input size
        ys_pd_array = sitk.GetArrayFromImage(ys_pd_sitk)
        ys_array = np.zeros((srcSize[2], srcSize[1], srcSize[0]), 'uint8')
        ys_array[0:np.shape(ys_pd_array)[0], 0:np.shape(ys_pd_array)[1], 0:np.shape(ys_pd_array)[2]] = ys_pd_array
        # step8 out put predict mask
        ys_array = ys_array.astype('uint8')
        ys_pd_itk = sitk.GetImageFromArray(ys_array)
        ys_pd_itk.SetSpacing(srcSpacing)
        ys_pd_itk.SetOrigin(srcOrigin)
        ys_pd_itk.SetDirection(srcDirection)
        out_mask_image = out_path + "/mask" + str(path_list[subsetindex])
        sitk.WriteImage(ys_pd_itk, out_mask_image)


if __name__ == "__main__":
    inference1mm()
