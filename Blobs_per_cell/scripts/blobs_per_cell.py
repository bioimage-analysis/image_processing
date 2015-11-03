import numpy as np
from skimage.morphology import dilation
from skimage.morphology import remove_small_objects
import numpy.ma as ma
from skimage.filters import sobel
from skimage.morphology import binary_closing
from skimage.morphology import square
from skimage.morphology import watershed
from skimage.measure import label
from skimage import feature
from skimage.filters import gaussian_filter
from skimage.morphology import reconstruction
from scipy import signal
from matplotlib.patches import Circle
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from skimage import io
import os
from matplotlib.patches import Circle
import csv
import time
from itertools import izip
from skimage.filters import threshold_otsu
import pandas as pd
from skimage.measure import regionprops

def number_nucleus(image):

    elevation_map = sobel(image)
    markers = np.zeros_like(image)
    markers[image < 250] = 1
    markers[image > 2000] = 2

    segmentation = watershed(elevation_map, markers)
    label_img = label(segmentation)
    prop = regionprops(label_img)

    width, height = plt.rcParams['figure.figsize']
    plt.rcParams['image.cmap'] = 'gray'

    image_label_overlay = label2rgb(label_img, image=image)

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 8))
    ax1.imshow(image_label_overlay)
    ax2.imshow(image, cmap=plt.cm.gray, interpolation='nearest')

    # create list of region with are < 1000
    image_labeled = [region for region in prop if region.area > 5000]


    return len(image_labeled)


def image_info(image):

    print(image.shape)
    print(image.dtype)
    print(image.size)
    print(image.ndim)
    print(np.amin(image))
    print(np.amax(image))


def image_processing(image, **kwargs):

    footprint_dic = {
    # Kernel "Laplacian" of size 3x3+1+1 with values from -1 to 8
    # Forming a output range from -8 to 8 (Zero-Summing)
    'footprint1': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
    'footprint2': np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,21,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]]),
    #Kernel "Laplacian" of size 5x5+2+2 with values from -2 to 16
    #Forming a output range from -16 to 16 (Zero-Summing)
    'footprint3': np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
    }

    footprint = kwargs.get('footprint', footprint_dic['footprint1'])

    image_conv = signal.convolve2d(image, footprint)
    image_conv = (image_conv.clip(min=0)).astype(np.uint16)
    image_conv = gaussian_filter(image_conv, sigma = 3)

    return image_conv


def blobs(image, remove_mb = None, val = 160, size = 100):
    """ Convolve a kernel on the image and a gaussian filter to highligh blobs. Find blobs using the
    Difference of Gaussian. Remove from the list of blobs the blobs that are at the membrane.
    return 3 different list
    """

    thresh = threshold_otsu(image)

    #Find all the blobs in the image using Difference of Gaussian
    blobs_in_image = feature.blob_dog(image, min_sigma=0.01,
                        max_sigma=3, threshold=thresh)
    blob_list = []
    for blob in blobs_in_image:
        y, x, r = blob
        blob_list.append((y, x))



    if remove_mb == None:
        blob_in_image_after_binary = set(blob_list)

    else:
        #Create a mask to remove blobs that are at the membrane and surrounded
        #by bright big object
        binary = image >= val*thresh/100
        binary = dilation(binary, square(3))
        binary = remove_small_objects(binary, min_size=size)
        # Create a list of coordinate with the binary image
        coor_binary = np.nonzero(binary)
        list_blob_masked = zip(*coor_binary)
        #Substract the list of coordinate from the binary image to the list of blobs
        blob_in_image_after_binary = (set(blob_list) - set (list_blob_masked))

    return blob_in_image_after_binary


def show_result (blob_in_image_after_binary, image, clim = (0.0, 2000)):
    """ Show which blob where selected, could adjust parameters in function
    "blobs_channel1_in_channel2" to get more or less blobs.
    """

    blobs_list = [blob_in_image_after_binary]
    colors = ['red']
    titles = ['Blobs']

    sequence = zip(blobs_list, colors, titles)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8))

    for blobs, color, title in sequence:

        ax1.imshow(image, cmap=plt.cm.gray, interpolation='nearest', clim=clim)
        for blob in blobs:
            y, x = blob
            c = plt.Circle((x, y), color=color, linewidth=2, fill=False)
            ax1.add_patch(c)
        ax2.imshow(image, cmap=plt.cm.gray, interpolation='nearest', clim=(0.0, 2000))




def batch_analysis(path):
    imageformat=".tif"
    imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
    image_blob = [io.imread(image) for image in imfilelist]

    return image_blob


def plot_result(paths, values, name_exp ='Ecad spot in', title = "Redcma", save = False):

    folder_names = []
    for path in paths:
        path, folder_name = os.path.split(path)
        folder_names.append(folder_name)

    keys = ['%s %s' % (name_exp, str(folder_name)) for folder_name in folder_names]

    results = dict(zip(keys, values))

    #filename = os.path.basename(path)

    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.transpose()
    df.plot(kind='box', title = title)

    if save == True:
        for i in range(len(paths)):
            filename = [filename for filename in os.listdir(paths[i]) if filename.endswith(('.nd2', '.tif'))]
            result_save = dict(zip(filename, values[i]))
            df_save = pd.Series(result_save)
            df_save = pd.DataFrame(df_save)

            counter = time.strftime("_%Y%m%d_%H%M")
            file = os.path.join(paths[i], folder_names[i] + str(counter) + ".xlsx")
            df_save.to_excel(file)
