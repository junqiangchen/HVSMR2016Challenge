import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
from Vnet.model_mutildepth_vnet3d_multilabel import mutildepthVnet3dModuleMultiLabel
import numpy as np
import pandas as pd




def trainmutildepthVnet():
    '''
    Vnet network segmentation kidney fine segmatation
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data/train1mm.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    Vnet3d = mutildepthVnet3dModuleMultiLabel(96, 96, 96, channels=1, numclass=3, costname=("categorical_focal_loss",))
    Vnet3d.train(imagedata, maskdata, "Vnet3d.pd", "log\\mutildepthVnet\\categorical_focal_loss\\", 0.001, 0.5, 10, 1,
                 [8, 12])


if __name__ == "__main__":
    trainmutildepthVnet()
