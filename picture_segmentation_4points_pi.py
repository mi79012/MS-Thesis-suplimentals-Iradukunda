# -*- coding: utf-8 -*-
"""
Created on Sun Jul 2 11:23:35 2023
"""

import cv2
import numpy as np
import os
from os import listdir, remove, mkdir
from os.path import isfile, join
from pandas import Interval
import matplotlib.pyplot as plt
import itertools
from datetime import datetime
from itertools import cycle

#for imagre manipulation and view
import PIL
from PIL import Image
import numpy
import pandas
# in pixels?
# h=2400, w=2300 2350, 2400
width = 2600#2350  # 1268
height = 2600 #2400  # 2460

mypath = r"/home/mark/Desktop/imgs"
mypath = mypath if mypath[-1] == "/" else mypath + "/"
# files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# playing folders
date_folders = [f for f in listdir(mypath) if os.path.isdir(join(mypath, f))]  # Get a list of date folders within img1

files = []
for date_folder in date_folders:
    folder_path = join(mypath, date_folder)
    if os.path.isdir(folder_path):  # Check if the path is a directory
        files_in_folder = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        files.extend([join(date_folder, f).replace("\\", "/") for f in files_in_folder])


# useful as there is movement of trays (makes tray flat)
def unwarp(img, src, dst, testing):  # src=source coordinates, dst=output coordinates
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

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
        
        # It was flipped, unflip it
        #ax2.imshow(warped)
        ax2.imshow(cv2.flip(warped, 1))
        ax2.set_title('Unwarped Image', fontsize=30)
        plt.show()
    else:
        return warped, M


# Calculating the start time of the experiment
times_from_epoch = []
for file in files:
    if file[-4:] == ".png":
        timestamp = file[11:30]
        try:
            time = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
            epoch_time = (time - datetime(2023, 1, 1)).total_seconds()
            if epoch_time not in times_from_epoch:
                times_from_epoch.append(epoch_time)
        except:
            #print("Weird file to evaluate the start time of the experiment!")
            pass
start_time = min(times_from_epoch)

tray_no = ""
for file in files:
    if file[-4:] == ".png":
        img = cv2.imread(mypath + file)  # open a file
        #img = img[70:2500, 1150:3700]  # img[200:2500 , 1100:3450]#img[70:2500 , 1150:3700]
        
        #convert image to RGB to allow PIL image manipulations (easier rotations, and cropping)
        image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image=Image.fromarray(img)
        #image=image.rotate(1.8)
        #image.crop((left, top, right, bottom))
        image = image.crop((800, 0, 3700, 2600)) #1000, 0, 3800, 2600 #1100, 100, 3550, 2540
        
        #img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #convert the image back to numpy array to allow cv2 operations
        img= numpy.asarray(image)
        
        width, height = img.shape[1], img.shape[0]
        try:
            #img = cv2.flip(img, 1)  # flp the image horizontally
            #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)#rotate to counter the flipping
            #img = cv2.rotate(img, cv2.ROTATE_180)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # change the color space

            lower = [117,143,78] #0, 152, 90 #was for study1 [0, 150, 205]# lower threshold for red color
            higher = [179, 255, 255] #6, 255, 255 #was for study1 [7, 255, 255]  # higher threshold for red color

            lower = np.array(lower, dtype="uint8")
            higher = np.array(higher, dtype="uint8")

            mask = cv2.inRange(img, lower, higher)

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

            vertical_segment = int((width - 0) / 2)
            horizontal_segment = int((height - 0) / 2)

            timestamp = file[11:30]
            time_of_picture_take = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
            epoch_time_of_picture_take = (time_of_picture_take - datetime(2023, 1, 1)).total_seconds()
            delay_from_start = int((epoch_time_of_picture_take - start_time) / 3600)

# =============================================================================
#             directory_name = join(mypath, date_folder, "Tray " + date_folder, file[11:47])
#             try:
#                 os.makedirs(directory_name)
#             except FileExistsError:
#                 pass
# =============================================================================

            #tray_path = join(mypath, date_folder, "Tray " + date_folder)
            # Add this line to create the tray_path folder for each date_folder
            # =============================================================================
# =============================================================================
#             for file in files:
#                 tray_path = join(mypath+ file[0:10]+"/"+"Tray "+file[0:10])
#                 os.makedirs(tray_path, exist_ok=True)
# =============================================================================
            # =============================================================================

            for pot_row in range(2):
                for pot_col in range(2):
                    try:
                        os.mkdir(mypath+ file[0:10]+"/"+"Tray "+file[0:10])
                    except:
                        pass

                    crop_image = warped[0 + horizontal_segment * (pot_col):0 + horizontal_segment * (pot_col + 1),
                                 0 + vertical_segment * (pot_row):0 + vertical_segment * (pot_row + 1)]

                    filename = f"{file[11:47]}_{pot_col + 1}-{pot_row + 1}.png"#_{delay_from_start}
                    #file_path = os.path.join(tray_path, filename)
                    file_path = os.path.join(mypath+ file[0:10]+"/"+"Tray "+file[0:10], filename)
                    cv2.imwrite(file_path, crop_image)

        except:
            #print("error " + str(file))
            pass

        tray_no = file[11:47]

#plt.imshow(warped)

#image = Image.open("/home/mark/Desktop/imgs/2023-08-22/2023-08-22-21-18-12_imx708_wide_rep1.png")
#try cropping using PIL
#image=image.rotate(1.8)
#image = image.crop((1000, 0, 3800, 2600))
#image.show()


#cmd F5 fro selected line
#image.crop((left, top, right, bottom))
#img=cv2.imread("/home/mark/Desktop/imgs/2023-07-18/2023-07-18-12-00-03_imx708_wide_rep1.png")
#img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# image=Image.fromarray(img)
# image=image.rotate(1.8)
# image = image.crop((1150, 150, 3450, 2400))
#image.show()
#[100:2400, 1100:3500]

#Signal the end
print("DONE")