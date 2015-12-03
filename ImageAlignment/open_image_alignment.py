import javabridge
import bioformats
from bioformats import log4j
from scipy.misc import imresize
import numpy as np
import os
import matplotlib.pyplot as plt
from textwrap import wrap
import math

JVM_BEGIN = False
JVM_END = False

plt.rcParams["axes.titlesize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

def begin_javabridge(max_heap_size='8G'):
    ''' Begin the jave virtual machine.

    Parameters
    ----------
    max_heap_size : string, optional
        Allocated memory for the virtual machine.

    Notes
    -----
    Remember to end the javabridge!
    '''

    global JVM_BEGIN

    javabridge.start_vm(class_path=bioformats.JARS,max_heap_size=max_heap_size)
    log4j.basic_config()

    JVM_BEGIN = True

def end_javabridge():

    ''' End the java virtual machine.

    Notes
    -----
    When killed, it cannot be restarted.
    '''

    global JVM_END

    javabridge.kill_vm()

    JVM_END = True

def image_info(image):
    ''' Extract interesting metadata from a sincle image (not to use with batch).

    Returns
    -----
    Dict with different parameters.
    '''

    if JVM_BEGIN == False:
        begin_javabridge()
    if  JVM_END == True:
        raise RuntimeError("The java virtual Machine has already ended"
                            "you should restart the program")
    else:
        with bioformats.ImageReader(image) as rdr:
            jmd = javabridge.JWrapper(rdr.rdr.getMetadataStore())

    if jmd.getPixelsPhysicalSizeX(0) == None:
        xsize = None
    else:
        xsize = javabridge.run_script('java.lang.Double(test)',
                                      bindings_in=dict(test = jmd.getPixelsPhysicalSizeX(0)))
    if jmd.getPixelsSizeC(0) == None:
        channels = None
    else:
        channels = javabridge.run_script('java.lang.Integer(test)',
                                         bindings_in=dict(test = jmd.getPixelsSizeC(0)))
    if jmd.getPixelsSizeT(0) == None:
        time_frames = None
    else:
        time_frames = javabridge.run_script('java.lang.Integer(test)',
                                            bindings_in=dict(test = jmd.getPixelsSizeT(0)))

    if time_frames <= 1:
        time_interval = None
    else:
        t0 = javabridge.run_script('java.lang.Double(test)',
                                   bindings_in=dict(test = jmd.getPlaneDeltaT(0, 0)))
        t1 = javabridge.run_script('java.lang.Double(test)',
                                   bindings_in=dict(test = jmd.getPlaneDeltaT(0, 3)))
        time_interval = round((t1 - t0),3)

    if jmd.getPixelsSizeZ(0) == None:
        z_steps = None
    else:
        z_steps = javabridge.run_script('java.lang.Integer(test)',
                                        bindings_in=dict(test = jmd.getPixelsSizeZ(0)))
    if jmd.getPixelsPhysicalSizeZ(0) == None:
        z_step_size = None
    else:
        z_step_size = javabridge.run_script('java.lang.Double(test)',
                                            bindings_in=dict(test = jmd.getPixelsPhysicalSizeZ(0)))

    if jmd.getPixelsSizeX(0)== None:
        frame_size_x = None
    else:
        frame_size_x = javabridge.run_script('java.lang.Double(test)',
                                            bindings_in=dict(test = jmd.getPixelsSizeX(0)))

    if jmd.getPixelsSizeY(0)== None:
        frame_size_y = None
    else:
        frame_size_y = javabridge.run_script('java.lang.Double(test)',
                                            bindings_in=dict(test = jmd.getPixelsSizeY(0)))

    return {
    "xsize" : xsize,
    "channels" : channels,
    "time_frames" : time_frames,
    "time_interval": time_interval,
    "z_steps" : z_steps,
    "z_step_size" : z_step_size,
    "frame_size_x" : frame_size_x,
    "frame_size_y" : frame_size_y
    }


def read_bioformat (image, resize = False):

    '''Read Images in almost any format.

    Parameters
    ----------
    resize : bool, optional
        If "true", will resize image to 1024, while keeping the ratio.

    Returns
    -------
    image : numpy ndarray, 5 dimensions
        The read image.
    '''

    if JVM_BEGIN == False:
        begin_javabridge()
    if  JVM_END == True:
        raise RuntimeError("The java virtual Machine has already ended "
                            "you should restart the program")

    else:
        with bioformats.ImageReader(image) as rdr:
            image = rdr.read(rescale=False)
            if image.dtype != 'uint16':
                image = rdr.read(c=0, rescale=False)

    if resize == True:
        size = np.max(image.shape)
        if size > 1024:
            image = imresize (image, 1024./size)

    return image

def batch_analysis_bioformat(path, **kwargs):

    """Go through evry image files in the directory (path).

    Parameters
    ----------
    path : str
    kwargs : dict
        Additional keyword-argument to be pass to the function:
         - imageformat

    """


    imageformat= kwargs.get('imageformat', '.nd2')

    imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]

    list_images = [read_bioformat(im, resize = True) for im in imfilelist]

    return list_images, imfilelist

def image_reorder(image):

    ''' Reorder images following scikit-image convention:
    (t, pln, row, col, ch)

    '''
    dic = image_info(image)
    x = int(dic['frame_size_x'])
    y = int(dic['frame_size_y'])
    c = int(dic['channels'])
    z = int(dic['z_steps'])
    t = int(dic['time_frames'])

    with bioformats.ImageReader(image, perform_init=True) as rdr:
        img = np.empty([t,z,y,x,c], np.uint16)

        for t in range(t):
            img[t,:,:,:,:] = rdr.read(t=t, rescale=False)
        for z in range(z):
            img[:,z,:,:,:] = rdr.read(z=z, rescale=False)
        for c in range(c):
            img[:,:,:,:,c] = rdr.read(c=c, rescale=False)

    return img

def scale_img(img, scale_min=None, scale_max=None):


	imageScale=np.array(img, copy=True)

	if scale_min == None:
		scale_min = imageScale.min()
	if scale_max == None:
		scale_max = (imageScale.max()*10)/100

	imageScale = imageScale.clip(min=scale_min, max=scale_max)
	imageScale = imageScale - scale_min
	indices = np.where(imageScale < 0)
	imageScale[indices] = 0.0
	imageScale = np.sqrt(imageScale)
	imageScale = imageScale / math.sqrt(scale_max - scale_min)

	return imageScale


def overlay_channels (im, scale_min=None, scale_max=None):

    try:
        im = read_bioformat(im)
    except AttributeError:
        pass

    if im.ndim == 3:
        r = im[...,0]
        g = im[...,1]
        img = np.empty((im.shape[0], im.shape[1], 3), np.uint16)
        img[...,0] = scale_img(r, scale_min=scale_min,scale_max=scale_max)
        img[...,1] = scale_img(g, scale_min=scale_min,scale_max=scale_max)

    elif im.ndim > 3:
        image = np.empty((im.shape[2], im.shape[3], 2), np.uint16)
        image[:,:,:] = im[0,0,:,:,:]
        r = image[...,0]
        g = image[...,1]
        img = np.empty((image.shape[0], image.shape[1], 3), np.uint16)
        img[...,0] = scale_img(r, scale_min=scale_min, scale_max=scale_max)
        img[...,1] = scale_img(g, scale_min=scale_min, scale_max=scale_max)


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 8))

    ax1.imshow(r, cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_title('Red channel')
    ax2.imshow(g, cmap=plt.cm.gray, interpolation='nearest')
    ax2.set_title('Green channel')
    ax3.imshow(img, interpolation='nearest')
    ax3.set_title('Green = Green channel, Red = Red channel')
    plt.show()
