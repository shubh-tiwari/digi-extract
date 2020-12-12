""""Module to extract handwritten digits and symbols of mathematical expression from image"""

import os
import numpy as np
from matplotlib import pyplot as plt
import cv2

def compare_cnt(rects): 
    """function to compare contours"""
    rects.sort(key = lambda x: x[0])  
    return rects 

def segment_reshape(segment, size = [28,28], pad = 0):
    """function for resize images"""
    m,n = segment.shape
    idx1 = list(range(0,m, (m)//(size[0]) ) )
    idx2 = list(range(0,n, n//(size[1]) )) 
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i,j] = segment[ idx1[i] + (m%size[0])//2, idx2[j] + (n%size[0])//2]
    return out

def extract_line(image):
    """function to extract written line area"""
    th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,251,1)

    kernel = np.ones((9,9),np.uint8)
    dilation = cv2.dilate(th3,kernel,iterations = 2)

    kernel = np.ones((11,51),np.uint8)
    erosion = cv2.erode(dilation,kernel,iterations = 5)

    contours, _ = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    line_area = 0
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        if h*w > 1000 and h*w < 0.9*image.shape[0]*image.shape[1]:
            if h*w > line_area:
                line_segment = image[y:y+h, x:x+w]
                line_area = h*w
            #segments.append(image[y:y+h, x:x+w])
    return line_segment

def extract_sym(line_segment):
    """function to extract digits from line area"""
    ret3,th4 = cv2.threshold(line_segment,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,5),np.uint8)
    erosion1 = cv2.erode(th4,kernel,iterations = 5)
    contours, _ = cv2.findContours(erosion1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    char_seg = []
    rectlist = []
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        if h*w > 10000 and h*w < 0.8*line_segment.shape[0]*line_segment.shape[1]:
            rectlist.append(rect)
    rectlist = compare_cnt(rectlist)
    for rect in rectlist:
        x,y,w,h = rect
        temp = line_segment[y:y+h, x:x+w]
        temp = segment_reshape(temp)
        char_seg.append(temp)
    return char_seg