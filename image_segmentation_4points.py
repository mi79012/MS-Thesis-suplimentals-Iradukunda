# -*- coding: utf-8 -*-
"""
Created on Sun Jul 2 11:23:35 2023
@author: Mark Iradukunda
"""
#import neccessary libraries (You might not need some of them) 
import cv2
import numpy as np
import os
from os import listdir, remove, mkdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import itertools
from datetime import datetime

#for image manipulation (PIL works well if you running this on raspberry pi)
import PIL
from PIL import Image
import numpy



width = 2600#2350  # 1268
height = 2600 #2400  # 2460

#save imgs in the same parent folder
mypath = r".\imgs"
mypath = mypath if mypath[-1] == "/" else mypath + "/"

# Get a list of date folders within imgs folder
date_folders = [f for f in listdir(mypath) if os.path.isdir(join(mypath, f))] 

#all image files within data folders
files = []
for date_folder in date_folders:
    folder_path = join(mypath, date_folder)
    if os.path.isdir(folder_path):  # Check if the path is a directory
        files_in_folder = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        files.extend([join(date_folder, f).replace("\\", "/") for f in files_in_folder])


# This function makes tray flat
# src=source coordinates, dst=output coordinates
def unwarp(img, src, dst, testing): 
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    
#test an image
    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img)
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=30)
        
        # for visualization
        ax2.imshow(cv2.flip(warped, 1))
        ax2.set_title('Unwarped Image', fontsize=30)
        plt.show()
    else:
        return warped, M


# Calculating the start time of the experiment (year-month-day-hr-min-sec)
times_from_epoch = []
for file in files:
    if file[-4:] == ".png":
        timestamp = file[11:30]#extract that information from image file names
        try:
            time = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
            epoch_time = (time - datetime(2023, 1, 1)).total_seconds()
            if epoch_time not in times_from_epoch:
                times_from_epoch.append(epoch_time)
        except:
            #print("Weird file to evaluate the start time of the experiment!")
            #in case there is an image that doesn't belong (different file name)
            pass
start_time = min(times_from_epoch)

#work on the model image (all images will worked on based on the following features)
tray_no = ""
for file in files:
    if file[-4:] == ".png":
        img = cv2.imread(mypath + file) # open a file
        
        #convert image to RGB to allow PIL image manipulations (easier rotations, and cropping)
        image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image=Image.fromarray(img)#to allow PIL based manipulations
        #image=image.rotate(1.8) #rotate if necessary
        #image.crop((left, top, right, bottom))
        image = image.crop((800, 0, 3700, 2600)) #adjust
    
        #convert the image back to numpy array to allow cv2 operations
        img= numpy.asarray(image)
        
        width, height = img.shape[1], img.shape[0]
        try:
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # change the color space to HSV
            
            #this values are based on HSV colors and they can identify red pins on the tray
            lower = [117,143,78] # lower threshold for red color
            higher = [179, 255, 255] #higher threshold for red color

            lower = np.array(lower, dtype="uint8")
            higher = np.array(higher, dtype="uint8")
            
            #mask image for the square created by 4 pins
            mask = cv2.inRange(img, lower, higher)
            
            #draw countours in that square (box) and get 4 centers
            cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            img = cv2.drawContours(img, cont, -1, (0, 0, 255), 1)

            if file[11:47] != tray_no:
                contours = []
                for c in cont:
                    x, y, w, h = cv2.boundingRect(c)
                    contours.append([w * h, [x, y]])
                contours = sorted(contours)
                contours = contours[-4:]
                centers = []
                for i in range(len(contours)):
                    centers.append(contours[i][1])
                centers = np.array(centers)

                sums_centers = []
                for e in centers:
                    sum = e[0] + e[1]
                    sums_centers.append((sum))

                zipped_lists = zip(sums_centers, centers)
                sorted_zipped_lists = sorted(zipped_lists)
                centers = [element for _, element in sorted_zipped_lists]
                centers = np.array(centers)

                x = []
                if len(centers) > 4:
                    for center in centers:
                        x.append(center[0])

                    x = sorted(x)
                    x = [x[0], x[1], x[-2], x[-1]]
                    x = np.array(x)
                    deleted = 0
                    for id, center in enumerate(centers):
                        if center[0] not in x[:]:
                            centers = np.delete(centers, id * 2 - deleted)
                            centers = np.delete(centers, id * 2 - deleted)
                            deleted += 2
                    centers = centers.reshape(4, 2)
                centers = np.float32(centers)

            dst = np.float32([(0, 0),
                              (0, height),
                              (width, 0),
                              (width, height)])

            warped, M = unwarp(img, centers, dst, False)
            warped = cv2.cvtColor(warped, cv2.COLOR_HSV2BGR)
            
            
            timestamp = file[11:30]
            time_of_picture_take = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
            epoch_time_of_picture_take = (time_of_picture_take - datetime(2023, 1, 1)).total_seconds()
            delay_from_start = int((epoch_time_of_picture_take - start_time) / 3600) #to seconds

            '''
            Divide the square box into 4 equal parts
            You can get other dimensions by channging denominator
            For example: 6 sections can be achived by setting (3,2 or 2,3)
            For 2 sections: set them to 1,1 and so on....
            '''
            
            vertical_segment = int((width - 0) / 2)
            horizontal_segment = int((height - 0) / 2)

            #These values must match those in vertical/horizontal segment
            for pot_row in range(2):
                for pot_col in range(2):
                    try:
                        os.mkdir(mypath+ file[0:10]+"/"+"Tray "+file[0:10])
                    except:
                        pass
                    
                    #Label each section: We moved from left to right, then top to bottom
                    #top left (1,1), top right(1,2), bottom left(2,1), bottom right(2,2)
                    #Python counts from 0, so 1,1 would be (0,0)
                    crop_image = warped[0 + horizontal_segment * (pot_col):0 + horizontal_segment * (pot_col + 1),
                                 0 + vertical_segment * (pot_row):0 + vertical_segment * (pot_row + 1)]
                    
                    #Each each section is cut and made into a new image with the following file names
                    filename = f"{file[11:47]}_{pot_col + 1}-{pot_row + 1}.png"#
                    
                    #Make a new folder (Tray_date) withing that date folder that contains new smaller images
                    #We want cut images but also must keep the original image.
                    file_path = os.path.join(mypath+ file[0:10]+"/"+"Tray "+file[0:10], filename)
                    cv2.imwrite(file_path, crop_image)

        except:
            #print("error " + str(file)) and continue to the next file
            pass

        tray_no = file[11:47]


#Signal the end
print("DONE")
