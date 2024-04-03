
"""
Created on Sun Jul 2 11:23:35 2023
@author: Mark Iradukunda
"""
#import neccessary libraries (packages and how to install them should available on the website of the vendor)
from picamera2 import Picamera2, Preview
from libcamera import controls #to help with focus controls
import time
#other libraries (from the old scripts/camera module2)
import RPi.GPIO as GPIO#to activate camera
import os
import shutil

#libraries for image manipulation and view
import PIL
from PIL import Image
#If one wants to let the camera show max settings (resolution)
from pprint import *

picam2 = Picamera2()#call the camera (model dependent)

pprint(picam2.sensor_modes)

#Configure the image
img_config = picam2.create_still_configuration() #default valeus
picam2.configure(img_config)
picam2.start()
#time.sleep(2)

#image directory
base_directory = "/home/mark/Desktop/imgs"
#make sure the device time is set to your timezone
current_date = time.strftime("%Y-%m-%d")#data format (year-mont-day)

# Create the folder for the current date. All images will be grouped for that day
date_folder = os.path.join(base_directory, current_date)
os.makedirs(date_folder, exist_ok=True)
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

#This will help if you have multiple camera's and you want to know what camera took them from the filename
#Additional information to help identify image/tray (eg: replication number)
camera_name=picam2.camera_properties['Model']+"_rep1" #get the camera model/info rep1 

#finalize file anme and path to save it
image_filename = f"{timestamp}_{camera_name}.png"
image_path = os.path.join(date_folder, image_filename)


#Establish the focus to infity (solve the bluryness of the close objects)
picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0})# Focus to infity

#leave some time to allow for configureations to establish
time.sleep(2)


########################
#check camera properties and configuration details (remove '#' to execute)
#picam2.camera_properties
#img_config
#picam2.sensor_modes
############################ 

#capture and save image
picam2.capture_file(image_path) #capture and store image

#If you want to open and view the captured image (remove '#')
#image = Image.open(image_path) #read image file
#image.show() #show image

#Signal end of execution
print("DONE")
