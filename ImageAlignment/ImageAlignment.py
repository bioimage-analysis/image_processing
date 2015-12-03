import cv2
import numpy as np
from skimage import img_as_ubyte
import open_image_bioformat_al as oib
import bioformats
import warnings
import os
from skimage.external import tifffile

warnings.filterwarnings("ignore", category=UserWarning, module='skimage')

def define_matrix(image, number_of_iterations = 5000, termination_eps = 1e-10, warp = 'Affine'):

    warp_mode_dct = {
    'Translation' : cv2.MOTION_TRANSLATION,
    'Affine' : cv2.MOTION_AFFINE,
    'Euclidean' : cv2.MOTION_EUCLIDEAN,
    'Homography' : cv2.MOTION_HOMOGRAPHY
    }

    img = oib.image_reorder(image)

    color1 = img_as_ubyte(img[0,0,:,:,0])
    color2 = img_as_ubyte(img[0,0,:,:,1])

    warp_mode = warp_mode_dct.pop('%s' % warp)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    number_of_iterations = number_of_iterations
    termination_eps = termination_eps
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC (color1,color2,warp_matrix, warp_mode, criteria)

    return warp_matrix

def align(im, warp_matrix):


    dic = oib.image_info(im)

    x = int(dic['frame_size_x'])
    y = int(dic['frame_size_y'])
    c = int(dic['channels'])
    z = int(dic['z_steps'])
    t = int(dic['time_frames'])

    with bioformats.ImageReader(im, perform_init=True) as rdr:
        image = np.empty([t,z,y,x,c], np.uint16)
        for c in range(c):
            image[:,:,:,:,c] = rdr.read(c=1, rescale=False)
            image[:,:,:,:,c] = cv2.warpAffine((rdr.read(c=0, rescale=False)),
                                               warp_matrix, (y,x), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            for t in range(t):
                image[t,:,:,:,c] = cv2.warpAffine((rdr.read(t=t, z=0, c=0, rescale=False)),
                                                   warp_matrix, (y,x), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                for z in range(z):
                    image[:,z,:,:,c] = cv2.warpAffine((rdr.read(z=z, c=0, rescale=False)),
                                                       warp_matrix, (y,x), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return image

def imsave_al(image, im, path):

    dic = oib.image_info(im)
    if dic['time_interval'] == None:
        t = 0
    else:
        t = float(dic['time_interval'])
    sx = (1/float(dic['xsize']))

    filename = [os.path.splitext(filename)[0] for filename in os.listdir(path) if filename.endswith(('.nd2'))]
    filename
    i = 0
    for img in image:
        i+=1
        filename_save = (path+ '/' + ('%s%i' %(filename[0], i)) + '.tif')
        filename_save
        if os.path.isfile(filename_save) == True:
            print 'the file already exist'
        else:
            tifffile.imsave(filename_save, img.transpose(0,1,4,2,3), imagej=True, resolution = (sx,sx), metadata = {'mode' : 'color', 'finterval' : t, 'unit' : 'micron'})
