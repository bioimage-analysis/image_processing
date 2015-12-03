## Repository overview

This repository contains few image processing module that I wrote, Blobs_per_cell and membrane_accumulation were used to do the analysis of a paper that was submited resently in JCS and ImageAlignment is use in the Stanford imaging facility. For all these modules I made a notebook in order to have a better understanding on how they work. 

1. All these modules includes an [open_image_bioformat](https://github.com/cespenel/image_processing/blob/master/Blobs_per_cell/scripts/open_image_bioformat.py) use python_bioformat package to allow to work with different image format, here is what it should be able to do:
 1. Beginning and ending the javabridge to work python_bioformat
 2. Create a dictionary with "important" image information (pixel size, number of channels etc.)
 3. Read the image and return a numpy array
 4. Do some batch analysis (go through a folder and return a list of numpy array)
 5. Show all or a chunk of your images (with or without histogram) that are located in your folder

2. [blobs_per_cells](https://github.com/cespenel/image_processing/blob/master/Blobs_per_cell/scripts/blobs_per_cell.py) use the scikit-image package to analyse the images:
 1. Extract the number of nucleus (number of cells)
 2. Do some image processing
 3. Extract the amount of blobs
 4. Circle the blobs it found
 5. Plot and save results as a box plot using pandas package

 6. It also includes [blobs_per_cells_click](https://github.com/cespenel/image_processing/blob/master/Blobs_per_cell/scripts/blobs_per_cell_click.py) is very similar to blobs_per_cell put can measure click events on the image:
 * Extract the number of cells by measuring click events
 * Do some image processing
 * Extract the amount of blobs
 * Circle the blobs it found
 * Plot and save results as a box plot using pandas package
 
 
3. [membrane_accumulation](https://github.com/cespenel/image_processing/blob/master/membrane_accumulation/scripts/segmentation_click.py) use the scikit-image package and the OpenCV package to analyse the images::
 1. Draw a region of interest
 2. Segment the accumulation in the ROI
 3. Measure the surface occupy by the ROI and the segmented region


4. [ImageAlignment](https://github.com/cespenel/image_processing/blob/master/ImageAlignment/ImageAlignment.py) use the OpenCV3 package to realign images::
 1. Determine a matrix that can then be applied to realign the images
 2. work with any format and with [ZTCYX] dimension
 3. Save images as a TIF file keeping some essential metadata

## Dependencies

This code uses a number of features in the scientific python stack. Thus far, this code has only been tested in a Mac OS environment, it may take some modification to run on other operating systems.

I highly recommend installing a scientific Python distribution such as Anaconda to handle the majority of the Python dependencies in this project.

###Python Dependencies

* [Numpy and Scipy](http://www.scipy.org/), numeric calculations and statistics in Python 
* [matplotlib](http://matplotlib.org/), plotting in Python
* [Pandas](http://pandas.pydata.org/), data-frames for Python, handles the majority of data-structures  
* [scikit-image](http://scikit-image.org/), used for image processing


