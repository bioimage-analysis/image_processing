import open_image_bioformat as oib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage import morphology
import os
import pandas as pd
import time


class MembraneAccumulation:

    def __init__(self, image):

        self.drawing = False
        self.image = image

        self.coords_cell = []
        self.surface_segmented = []
        self.surface_masked = []

    def click_canvas(self):
        self.fig, (self.ax1) = plt.subplots(1,1, figsize=(15, 8))
        self.ax1.imshow(self.image, cmap=plt.cm.gray, interpolation='nearest')

        # Connection to 3 different events, left click, motion and key press
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.click_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.press)

        plt.show()

    def onclick(self, event):

        # Will Create a left click event:
        # Get the coordinate of the event and draw a circle.

        if event.button == 1:
            self.drawing = True
            self.coords_cell = [(event.xdata, event.ydata)]

    def click_release(self, event):
        self.drawing = False

    def on_motion(self, event):

        # When click left, will append the coordinate of the mouse while mouving
        # it and draw circles until button is not press anymore.

        if self.drawing == True:
            self.press = self.coords_cell.append((event.xdata, event.ydata))
            circ = Circle((event.xdata, event.ydata), 5, color = 'yellow')
            self.ax1.add_artist(circ)
            self.fig.canvas.draw()

    def press(self, push):

        # Keyboard avent:
        # When press q, the figure will close.

        if push.key == 'q':
            plt.close(self.fig)

        # When press c:
        # ==> Image will be process, segmented and the surface segmented
        # and masked will be measured.

        if push.key == 'c':
            self.CleanImgage()
            self.ImageSegmentation()
            #plt.close(self.fig)
            self.show_result()

        if push.key == 'd':
            self.CleanImgage()
            self.segmentation_crop_area()
            plt.close(self.fig)
            self.show_result()

    def CleanImgage(self, thresh = 30):

        # Image processing

        footprint = np.array([[-1,-1,-1],[-1,8,-1], [-1,-1,-1]])
        self.clean_image = cv2.medianBlur(self.image, 5)
        self.clean_image = cv2.filter2D(self.clean_image,-1,footprint)
        self.clean_image = cv2.medianBlur(self.clean_image, 5)
        self.markers = np.zeros_like(self.image)
        self.markers[self.clean_image < threshold_otsu(self.image)] = 1
        self.markers[self.clean_image >= ((threshold_otsu(self.image)*thresh)/100)] = 2
        self.markers[self.clean_image >= ((threshold_otsu(self.image)*50)/100)] = 3

    def ImageSegmentation(self):

        kernel = np.array(self.coords_cell, np.int32)
        circle = np.zeros(self.image.shape[:2], np.uint8)

        # link with polylines the coordinates of "left click", thickness could be adjusted,
        # Could also fill inside the polyline
        cv2.polylines(circle,[kernel],False,(255,0,0), thickness=5)
        kernel2 = np.array(self.coords_cell, np.int32)
        circle2 = np.zeros(self.image.shape[:2], np.uint8)
        cv2.polylines(circle2,[kernel2],False,(255,0,0), thickness=4)

        # Segmentation of the protein accumulation using watershed
        self.segmentation = morphology.watershed(self.clean_image, self.markers, mask = circle)
        self.segmentation[self.segmentation < 1.5] = 0
        self.segmentation = self.segmentation.astype('uint8')

        # Find contour of the segmented area
        contours,hierarchy = cv2.findContours(self.segmentation, 1, 2)

        # Find countour of the masked area
        contours_circle,hierarchy = cv2.findContours(circle2, 1, 2)
        self.area = [cv2.contourArea(cnt) for cnt in contours if (cv2.contourArea(cnt))!=0.0]

        self.area = sum(self.area)
        self.area_mask = [cv2.contourArea(cnt_cell) for cnt_cell in contours_circle]
        self.area_mask = sum(self.area_mask)

        if self.area > 0:
            self.surface_segmented.append(self.area)
        if self.area_mask > 0:
            self.surface_masked.append(self.area_mask)

    def segmentation_crop_area(self):

        kernel = np.array(self.coords_cell, np.int32)
        circle = np.zeros(self.image.shape[:2], np.uint8)
        cv2.polylines(circle,[kernel],False,(255,0,0), thickness=10)
        crop_img = cv2.bitwise_and(self.image, self.image, mask=circle)
        kernel2 = np.array(self.coords_cell, np.int32)
        circle2 = np.zeros(self.image.shape[:2], np.uint8)
        cv2.polylines(circle2,[kernel2],False,(255,0,0), thickness=2)
        self.segmentation = morphology.watershed(self.clean_image, self.markers, mask = circle)
        self.segmentation[self.segmentation < 1.5] = 0
        self.segmentation = segmentation.astype('uint8')
        contours,hierarchy = cv2.findContours(segmentation, 1, 2)
        contours_circle,hierarchy = cv2.findContours(circle2, 1, 2)
        self.area = [cv2.contourArea(cnt) for cnt in contours if (cv2.contourArea(cnt))!=0.0]
        self.area = sum(self.area)
        self.area_mask = [cv2.contourArea(cnt_cell) for cnt_cell in contours_circle]
        self.area_mask = sum(self.area_mask)

    def show_result(self):

        fig, (ax1) = plt.subplots(figsize=(15, 8))
        ax1.imshow(self.image, cmap=plt.cm.gray, interpolation='nearest')
        ax1.contour(self.segmentation, [1.5], linewidths=1.2, colors='y')
        ax1.axis('off')
        plt.show()

    def result(self):

        ratio = (sum(self.surface_segmented) / sum(self.surface_masked)) * 100

        result={
        'Surface Segmented' : self.surface_segmented,
        'Surface masked' : self.surface_masked,
        'ratio' : ratio
        }

        return self.surface_segmented, self.surface_masked, ratio



def plot_result(path, results, save = False, **kwargs):

    path, folder_name = os.path.split(path)

    print folder_name

    title= kwargs.get('title', folder_name)
    df = pd.DataFrame(results)

    df.plot(kind='box', title = title)
    plt.show()

    if save == True:
        counter = time.strftime("_%Y%m%d_%H%M")
        file = os.path.join(path, title + str(counter) + ".xlsx")
        df.to_excel(file)
