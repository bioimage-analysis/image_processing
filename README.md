## Repository overview

This repository contains 2 main modules, open_image_bioformat and blobs_per_cells, it was originally use to determine the amount of "blobs" per cells in different conditions ([click here for more info](https://github.com/cespenel/Image_processing/blob/master/HOW%20TO%20USE%20%22open_image_bioformat%22%20and%20%22blobs_per_cell%22%20module.ipynb)).

1. open_image_bioformat use python_bioformat package to allow to work with different image format, here is what it should be able to do:
 1. Beginning and ending the javabridge to work python_bioformat
 2. Create a dictionary with "important" image information (pixel size, number of channels etc.)
 3. Read the image and return a numpy array
 4. Do some batch analysis (go through a folder and return a list of numpy array)
 5. Show all or a chunk of your images (with or without histogram) that are located in your folder

2. blobs_per_cells use scikit-image package analyse the images:
 1. Extract the number of nucleus (number of cells)
 2. Do some image processing
 3. Extract the amount of blobs
 4. Circle the blobs it found
 5. Plot and save results as a box plot using pandas package

## Dependencies

This code uses a number of features in the scientific python stack. Thus far, this code has only been tested in a Mac OS environment, it may take some modification to run on other operating systems.

I highly recommend installing a scientific Python distribution such as Anaconda to handle the majority of the Python dependencies in this project.

###Python Dependencies

* [Numpy and Scipy](http://www.scipy.org/), numeric calculations and statistics in Python 
* [matplotlib](http://matplotlib.org/), plotting in Python
* [Pandas](http://pandas.pydata.org/), data-frames for Python, handles the majority of data-structures  
* [scikit-image](http://scikit-image.org/), used for image processing


