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
import open_image_bioformat as oib


def define_ch1_ch2(image, im, channel1 = "channel2", channel2 = "channel0"):
    dct = oib.image_info(im[0])
    num_channel = dct["channels"]
    channels = ['channel{0}'.format(x) for x in range (num_channel)]
    channel_lst ={}

    for x in range(len(channels)):
        channel_lst[channels[x]] = image[:,:,x]

    channel1 = channel_lst.get(channel1)
    channel2 = channel_lst.get(channel2)

    return channel1, channel2

def mask_numpy_array(channel1, channel2, thresh_o = 20):

    channel2_Otsu = threshold_otsu(channel2)
    channel2_thresh = channel2 > ((channel2_Otsu*thresh_o)/100)
    channel2_thresh = binary_closing(channel2_thresh, square(5))
    channel1_channel2_masked = ma.masked_array(channel1, mask=~channel2_thresh)
    channel1_channel2 = channel1_channel2_masked.filled(0)
    return channel1_channel2, channel2_thresh

class numbercell:

    coords_cell = []
    coords_cell_inj = []

    def __init__(self, channel1, channel2_thresh):

        self.coords_cell = []
        self.coords_cell_inj = []

        self.fig, (self.ax1, self.ax2) = plt.subplots(1,2, figsize=(15, 8))

        self.ax1.imshow(channel1, cmap=plt.cm.gray, interpolation='nearest')
        self.ax2.imshow(channel2_thresh, cmap=plt.cm.gray, interpolation='nearest')
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.show()

    def onclick(self, event):

        if event.button == 1:
            self.coords_cell.append((event.xdata, event.ydata))
            circ = Circle((event.xdata, event.ydata), 10, color = 'yellow')
            self.ax1.add_artist(circ)
            self.fig.canvas.draw()

        elif event.button == 3:
            self.coords_cell_inj.append((event.xdata, event.ydata))
            circ_inj = Circle((event.xdata, event.ydata), 10, color = 'red')
            self.ax2.add_artist(circ_inj)
            self.fig.canvas.draw()

    def press(self, close):
        if close.key == 'q':
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close(self.fig)

    def getnumbcells(self):

        self.number_of_cells_inj = len(self.coords_cell_inj)
        self.number_of_cells_tot = len(self.coords_cell)

        return self.number_of_cells_inj, self.number_of_cells_tot

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

def blobs(image, channel1_channel2, remove_mb = False, val = 160, size = 100):
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


    if remove_mb == False:
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

    coor_channel1_channel2 = np.nonzero(channel1_channel2)
    list_blob = zip(*coor_channel1_channel2)

    blob_only_in_channel2 = blob_in_image_after_binary & set(list_blob)
    blob_not_in_channel2 = blob_in_image_after_binary - set(blob_only_in_channel2)

    return blob_in_image_after_binary, blob_only_in_channel2, blob_not_in_channel2


def show_result (blob_in_image_after_binary, blob_only_in_channel2, blob_only_not_in_channel2, image):
    """ Show which blob where selected, could adjust parameters in function
    "blobs_channel1_in_channel2" to get more or less blobs.
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize=(15, 8))

    ax1.imshow(image, cmap=plt.cm.gray, interpolation='nearest')

    for blob in blob_in_image_after_binary:
        y, x = blob
        c = plt.Circle((x, y), color='red', linewidth=2, fill=False)
        ax1.add_patch(c)
    ax2.imshow(image, cmap=plt.cm.gray, interpolation='nearest')

    ax3.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    for blob in blob_only_in_channel2:
        y, x = blob
        c = plt.Circle((x, y), color='red', linewidth=2, fill=False)
        ax3.add_patch(c)
    ax4.imshow(image, cmap=plt.cm.gray, interpolation='nearest')


    fig.set_tight_layout(True)


def batch_analysis(path):
    imageformat=".tif"
    imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
    image_blob = [io.imread(image) for image in imfilelist]

    return image_blob


def plot_result(path, results, save = False, **kwargs):

    path, folder_name = os.path.split(path)

    title= kwargs.get('title', folder_name)
    df = pd.DataFrame(results)

    df.plot(kind='box', title = title)
    plt.show()

    if save == True:
        counter = time.strftime("_%Y%m%d_%H%M")
        file = os.path.join(path, title + str(counter) + ".xlsx")
        df.to_excel(file)
