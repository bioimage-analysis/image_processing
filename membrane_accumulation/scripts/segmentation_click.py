import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage import morphology
import time
import os
import csv
from itertools import izip
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

class MembraneAccumulation:

    coords_cell = []
    surface_segmented = []
    surface_masked = []

    def __init__(self, image):

        self.image = image

        self.coords_cell = []
        self.surface_segmented = []
        self.surface_masked = []

        self.fig, (self.ax1) = plt.subplots(1,1, figsize=(15, 8))
        self.ax1.imshow(image, cmap=plt.cm.gray, interpolation='nearest')

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.press)

        plt.show()

    def onclick(self, event):

        if event.button == 1:
            self.press = event.xdata, event.ydata
            circ = Circle((event.xdata, event.ydata), 10, color = 'yellow')
            self.ax1.add_artist(circ)
            self.fig.canvas.draw()

    def on_motion(self, event):

        if event.button == 1:
            self.press = self.coords_cell.append((event.xdata, event.ydata))
            circ = Circle((event.xdata, event.ydata), 10, color = 'yellow')
            self.ax1.add_artist(circ)
            self.fig.canvas.draw()

    def press(self, push):

        if push.key == 'q':
            plt.close(self.fig)

        if push.key == 'c':
            self.clean_image()
            self.segmentation()
            plt.close(self.fig)
            self.show_result()
            if self.area > 0:
                self.surface_segmented.append(self.area)
            if self.area_mask > 0:
                self.surface_masked.append(self.area_mask)

        if push.key == 'd':
            self.clean_image()
            self.segmentation_crop_area()
            plt.close(self.fig)
            self.show_result()

    def clean_image(self, thresh = 30):

        footprint = np.array([[-1,-1,-1],[-1,8,-1], [-1,-1,-1]])
        self.clean_image = cv2.medianBlur(self.image, 5)
        self.clean_image = cv2.filter2D(self.clean_image,-1,footprint)
        self.clean_image = cv2.medianBlur(self.clean_image, 5)
        self.markers = np.zeros_like(self.image)
        self.markers[self.clean_image < threshold_otsu(self.image)] = 1
        self.markers[self.clean_image >= ((threshold_otsu(image)*thresh)/100)] = 2

    def segmentation(self):

        kernel = np.array(self.coords_cell, np.int32)
        circle = np.zeros(self.image.shape[:2], np.uint8)
        cv2.polylines(circle,[kernel],False,(255,0,0), thickness=10)
        kernel2 = np.array(self.coords_cell, np.int32)
        circle2 = np.zeros(self.image.shape[:2], np.uint8)
        cv2.polylines(circle2,[kernel2],False,(255,0,0), thickness=2)
        self.segmentation = morphology.watershed(self.clean_image, self.markers, mask = circle)
        self.segmentation[self.segmentation < 1.5] = 0
        self.segmentation = self.segmentation.astype('uint8')
        contours,hierarchy = cv2.findContours(self.segmentation, 1, 2)
        contours_circle,hierarchy = cv2.findContours(circle2, 1, 2)
        self.area = [cv2.contourArea(cnt) for cnt in contours if (cv2.contourArea(cnt))!=0.0]
        self.area = sum(self.area)
        self.area_mask = [cv2.contourArea(cnt_cell) for cnt_cell in contours_circle]
        self.area_mask = sum(self.area_mask)

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

        return self.surface_segmented, self.surface_masked
