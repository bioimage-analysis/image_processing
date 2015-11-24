import javabridge
import bioformats
from bioformats import log4j
from scipy.misc import imresize
import numpy as np
import os
import matplotlib.pyplot as plt
from textwrap import wrap

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


    if jmd.getPixelsTimeIncrement(0) == None:
        time_interval = None
    else:
        time_interval = javabridge.run_script('java.lang.Integer(test)',
                                              bindings_in=dict(test = jmd.getPixelsSizeT(0)))
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

    dic = image_info(image)

    x = int(dic['frame_size_x'])
    y = int(dic['frame_size_y'])
    c = int(dic['z_steps'])

    with bioformats.ImageReader(image, perform_init=True) as rdr:
        img = np.empty([y,x,c], np.uint16)

        for z in range(c):
            img[:,:,z] = rdr.read(z=z, rescale=False)
    return img


def show_series_all (images, path, im, channel = 'channel0', **kwargs):

    """Plot all the images in the directory (path) with the name of the file.

    Parameters
    ----------

    images : ndarray
    path : str
    channel: str

    kwargs : dict
        Additional keyword-argument to be pass to the function:
         - imageformat
         - titles
         - size_fig

    """

    dct = image_info(im[0])

    if int(dct['channels']) >= 1:
        num_channel = dct["channels"]
        channels = ['channel{0}'.format(x) for x in range (num_channel)]
        channel_lst ={}
    else:
        if int(dct['z_steps']) > 1:
            del dct['channels']
            dct['channels'] = dct.pop('z_steps')
            num_channel = dct["channels"]
            channels = ['channel{0}'.format(x) for x in range (num_channel)]
            channel_lst ={}
        else:
            pass

    nrows = np.int(np.ceil(np.sqrt(len(images))))
    ncols = np.int(len(images) // nrows +1 )
    imageformat= kwargs.get('imageformat', '.tif')
    filename=[f for f in os.listdir(path) if f.endswith(imageformat)]
    titles = kwargs.pop('titles', filename)
    width, size = kwargs.get('size_fig', (5*ncols, 5*nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, size))

    for img, n, label, ax in zip(images, range(len(images)), titles, axes.ravel()):
        try:
            for x in range(len(channels)):
                channel_lst[channels[x]] = img[:,:,x]

            ch = channel_lst.get(channel)
            i = n // ncols
            j = n % ncols
            axes[i, j].imshow(ch,
                              interpolation='nearest', cmap='gray')
            ax.set_title("\n".join(wrap(str(label), width=20)))

        except IndexError:
            channel_lst[channels[0]] = img
            ch = channel_lst.get('channel0')
            i = n // ncols
            j = n % ncols
            axes[i, j].imshow(ch,
                              interpolation='nearest', cmap='gray')
            ax.set_title("\n".join(wrap(str(label), width=20)))

    for ax in axes.ravel():

        if not (len(ax.images)):
            fig.delaxes(ax)

    fig.set_tight_layout(True)

    #plt.show()

    #return len(axes.ravel()), nrows * ncols



def show_series_all_histo (images, im, channel = 'channel0', **kwargs):

    """Plot all the images in the directory (path) and their histogram
    side by side.

    Parameters
    ----------
    images : ndarray
    path : str
    channel: str
    kwargs : dict
        Additional keyword-argument to be pass to the function:
         - size_fig

    """
    dct = image_info(im[0])
    if int(dct['channels']) >= 1:
        num_channel = dct["channels"]
        channels = ['channel{0}'.format(x) for x in range (num_channel)]
        channel_lst ={}
    else:
        if int(dct['z_steps']) > 1:
            del dct['channels']
            dct['channels'] = dct.pop('z_steps')
            num_channel = dct["channels"]
            channels = ['channel{0}'.format(x) for x in range (num_channel)]
            channel_lst ={}
        else:
            pass

    plt.rcParams["xtick.labelsize"] = 5
    nrows = np.int(np.ceil(np.sqrt(len(images))))
    ncols = np.int(len(images) // nrows+1)
    width, size = kwargs.get('size_fig', (6*ncols, 2*nrows))

    fig, axes = plt.subplots(nrows, ncols*2, figsize=(width, size))

    for img, n in zip(images, range(len(images))):
        try:
            for x in range(len(channels)):
                channel_lst[channels[x]] = img[:,:,x]
            ch = channel_lst.get(channel)
            i = n // ncols
            j = n % ncols * 2
            axes[i, j].imshow(ch,
                              interpolation='nearest', cmap='gray')
            axes[i, j+1].hist(ch.ravel(),
                              log=True, bins=500, range=(0, img[:, :, 0].max()))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j+1].set_yticks([])

        except IndexError:
            channel_lst[channels[0]] = img
            ch = channel_lst.get('channel0')
            i = n // ncols
            j = n % ncols * 2
            axes[i, j].imshow(ch,
                              interpolation='nearest', cmap='gray')
            axes[i, j+1].hist(ch.ravel(),
                              log=True, bins=500, range=(0, img.max()))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j+1].set_yticks([])


    for ax in axes.ravel():
        if not (len(ax.patches)) and not (len(ax.images)):
            fig.delaxes(ax)

    fig.set_tight_layout(True)
    #plt.show()
    #return len(axes.ravel()), nrows * ncols


def show_chunk_series_all_histo(images, path, im, **kwargs):

    """Plot all the images in the directory (path) and their histogram
    side by side but allow to "chunk" the amount of images if too many.

    Parameters
    ----------
    images : ndarray
    path : str
    kwargs : dict
        Additional keyword-argument to be pass to the function:
         - size_chunk

    """

    def _split_seq(seq, size):
            newseq = []
            splitsize = 1.0/size*len(seq)
            for i in range(size):
                    newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
            return newseq

    size_chunk = kwargs.get('size_chunk', 4)

    if size_chunk >= 3:
        chunks = _split_seq(images, ((len(images)/size_chunk)+1))

        for series in chunks:
            show_series_all_histo(series, im)

    elif size_chunk == 2:
        chunks = _split_seq(images, ((len(images)/size_chunk)))

        for series in chunks:
            show_series_all_histo(series, im)

    else:
        width, size = kwargs.get('size_fig', (15,5))
        for img in images:
            fig, (ax_img, ax_histo) = plt.subplots(ncols=2, figsize=(width, size))
            ax_img.imshow(img[:,:,0],cmap=plt.cm.gray, interpolation='nearest')
            ax_histo.hist(img[:, :, 0].ravel(),log=True, bins=500, range=(0, img[:, :, 0].max()))
            #plt.show()


def split_channels (image, path, im, **kwargs):

    """Split all the channels

    Parameters
    ----------
    images : ndarray
    path : str

    Returns
    ----------



    """

    dct = image_info(im[0])
    num_channel = dct["channels"]
    channels = ['channel{0}'.format(x) for x in range (num_channel)]
    channel_lst ={}


    channel_lst ={}
    for x in range(num_channel):
        channel_lst[channels[x]] = image[:,:,x]

    nrows = np.int(np.ceil(np.sqrt(len(channel_lst))))
    ncols = np.int(len(channel_lst) // nrows)
    #imageformat= kwargs.get('imageformat', '.tif')
    #filename=[f for f in os.listdir(path) if f.endswith(imageformat)]
    titles = kwargs.pop('titles', 'x')
    width, size = kwargs.get('size_fig', (5*ncols, 5*nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, size))

    for label, n, ax in zip(sorted(channel_lst.keys()), range(len(channel_lst)), axes.ravel()):
        i = n // ncols
        j = n % ncols
        axes[i, j].imshow(channel_lst[label],
                          interpolation='nearest', cmap='gray')
        ax.set_title("\n".join(wrap(str(label), width=20)))

    for ax in axes.ravel():

        if not (len(ax.images)):
            fig.delaxes(ax)

    fig.set_tight_layout(True)
    #plt.show()
