from __future__ import print_function, division
import numpy as np
import SimpleITK as sitk
import random
from dataprocess.utils import file_name_path


def resize_image_itk(itkimage, newSpacing, originSpcaing, resamplemethod=sitk.sitkNearestNeighbor, flag=True):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    newSpacing = np.array(newSpacing, float)
    # originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originSpcaing
    newSize = originSize / factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    if resamplemethod == sitk.sitkNearestNeighbor and flag:
        itkimgResampled = sitk.Threshold(itkimgResampled, 0, 1.0, 255)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled


def subimage_generator_random(image, mask, patch_block_size, subnumbers=1000):
    """
    generate the sub images and masks with patch_block_size
    :param image:
    :param patch_block_size:
    :param stride:
    :return:
    """
    width = np.shape(image)[1]
    height = np.shape(image)[2]
    imagez = np.shape(image)[0]
    block_width = np.array(patch_block_size)[1]
    block_height = np.array(patch_block_size)[2]
    blockz = np.array(patch_block_size)[0]
    stridewidth = width - block_width
    strideheight = height - block_height
    stridez = imagez - blockz
    # step 1:if stridez is bigger 1,return  subnumbers
    if stridez >= 1 and stridewidth >= 1 and strideheight >= 1:
        x_min, x_max = block_width, width - block_width
        y_min, y_max = block_height, height - block_height
        z_min, z_max = blockz, imagez - blockz
        if z_min > z_max:
            z_min, z_max = z_max, z_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        hr_samples_list = []
        hr_mask_samples_list = []
        for index in range(subnumbers):
            centerx = random.randint(x_min, x_max)
            centery = random.randint(y_min, y_max)
            centerz = random.randint(z_min, z_max)
            centerz_min = centerz - blockz // 2
            centerz_max = centerz + blockz // 2
            centerx_min = centerx - block_width // 2
            centerx_max = centerx + block_width // 2
            centery_min = centery - block_height // 2
            centery_max = centery + block_height // 2
            if centerx_min < 0:
                centerx_min = 0
                centerx_max = centerx_min + block_width
            if centerx_max > width:
                centerx_max = width
                centerx_min = width - block_width
            if centery_min < 0:
                centery_min = 0
                centery_max = centery_min + block_height
            if centery_max > height:
                centery_max = height
                centery_min = height - block_height
            if centerz_min < 0:
                centerz_min = 0
                centerz_max = centerz_min + blockz
            if centerz_max > imagez:
                centerz_max = imagez
                centerz_min = imagez - blockz
            if np.max(mask[centerz_min:centerz_max, centerx_min:centerx_max, centery_min:centery_max]) != 0:
                hr_samples_list.append(image[centerz_min:centerz_max, centerx_min:centerx_max, centery_min:centery_max])
                hr_mask_samples_list.append(
                    mask[centerz_min:centerz_max, centerx_min:centerx_max, centery_min:centery_max])
        hr_samples = np.array(hr_samples_list).reshape((len(hr_samples_list), blockz, block_width, block_height))
        hr_mask_samples = np.array(hr_mask_samples_list).reshape(
            (len(hr_mask_samples_list), blockz, block_width, block_height))
        return hr_samples, hr_mask_samples
    # step 2:other sutitation,return one samples
    else:
        nb_sub_images = 1 * 1 * 1
        hr_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float)
        hr_mask_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float)
        rangz = min(imagez, blockz)
        rangwidth = min(width, block_width)
        rangheight = min(height, block_height)
        hr_samples[0, 0:rangz, 0:rangwidth, 0:rangheight] = image[0:rangz, 0:rangwidth, 0:rangheight]
        hr_mask_samples[0, 0:rangz, 0:rangwidth, 0:rangheight] = mask[0:rangz, 0:rangwidth, 0:rangheight]
        return hr_samples, hr_mask_samples


def make_patch(image, mask, patch_block_size, subnumbers):
    """
    make number patch
    :param image:[depth,512,512]
    :param patch_block: such as[64,128,128]
    :return:[samples,64,128,128]
    """
    image_subsample, mask_subsample = subimage_generator_random(image=image, mask=mask,
                                                                patch_block_size=patch_block_size,
                                                                subnumbers=subnumbers)
    return image_subsample, mask_subsample


def gen_image_mask(srcimg, segimg, index, shape, subnumbers, trainImage, trainMask):
    """
    :param img:
    :param segimg:
    :param index:
    :param shape:
    :param subnumbers:
    :param trainImage:
    :param trainMask:
    :return:
    """
    # step 1 get subimages (numberxy*numberxy*numberz,96, 96, 96)
    sub_srcimages, sub_maskimages = make_patch(srcimg, segimg, patch_block_size=shape, subnumbers=subnumbers)
    # step 2 only save subimages (numberxy*numberxy*numberz,96, 96, 96)
    samples, imagez, height, width = np.shape(sub_srcimages)[0], np.shape(sub_srcimages)[1], \
                                     np.shape(sub_srcimages)[2], np.shape(sub_srcimages)[3]
    for j in range(samples):
        sub_masks = sub_maskimages.astype(np.uint8)
        filepath1 = trainImage + "\\" + str(index) + "_" + str(j) + ".npy"
        filepath = trainMask + "\\" + str(index) + "_" + str(j) + ".npy"
        np.save(filepath1, sub_srcimages[j, :, :, :])
        np.save(filepath, sub_masks[j, :, :, :])


def normalize(slice, bottom=95, down=5):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        # tmp[tmp == tmp.min()] = -9
        return tmp


def preparesampling3dtraindata(heart_imagepath, heart_labelpath, trainImage, trainMask, shape=(16, 256, 256),
                               subnumbers=1000):
    mask_path_list = file_name_path(heart_labelpath, False, True)
    image_path_list = file_name_path(heart_imagepath, False, True)
    newSpacing = (1.0, 1.0, 1.0)
    for subsetindex in range(len(image_path_list)):
        # step1 load src image with window center and window level,then resize to new Spacing
        file_image = heart_imagepath + "/" + str(image_path_list[subsetindex])
        src = sitk.ReadImage(file_image, sitk.sitkInt16)
        _, src = resize_image_itk(src, newSpacing, src.GetSpacing(), sitk.sitkLinear)
        srcimg = sitk.GetArrayFromImage(src)
        # step2 load mask image ,then resize to new Spacing
        mask_path = heart_labelpath + "/" + str(mask_path_list[subsetindex])
        seg = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        _, seg = resize_image_itk(seg, newSpacing, seg.GetSpacing(), sitk.sitkNearestNeighbor, False)
        segimg = sitk.GetArrayFromImage(seg)
        # step 3 normalize image
        srcimg = normalize(srcimg)
        # step 4 get subimages and submasks
        gen_image_mask(srcimg, segimg, subsetindex, shape=shape, subnumbers=subnumbers, trainImage=trainImage,
                       trainMask=trainMask)


def preparetraindata():
    """
    :return:
    """
    heart_path = "E:\MedicalData\HVSMR2016\cropedHeart"
    image_name = "Training dataset"
    mask_name = "Ground truth"
    heart_imagepath = heart_path + "/" + image_name
    heart_labelpath = heart_path + "/" + mask_name
    trainImage = "E:\MedicalData\HVSMR2016\\traindata\Image"
    trainMask = "E:\MedicalData\HVSMR2016\\traindata\Mask"
    preparesampling3dtraindata(heart_imagepath, heart_labelpath, trainImage, trainMask, (96, 96, 96), 1000)


if __name__ == "__main__":
    preparetraindata()
