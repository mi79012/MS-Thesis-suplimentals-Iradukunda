from picamera2 import Picamera2, Preview
from libcamera import controls #to help with focus controls
import time
picam2 = Picamera2()

#other libraries (from the old scripts/camera module2)
import RPi.GPIO as GPIO
import os
import shutil

#fro image manipulation and view
import PIL
from PIL import Image
#If one wants to let the camera show max settings (resolution)
from pprint import *
pprint(picam2.sensor_modes)

#camera preview
#camera_config = picam2.create_preview_configuration()
#img configuration
#img_config = picam2.create_still_configuration()
#picam2.configure(img_config)
#picam2.start_preview(Preview.QT) #for a preview (silence it to allow cron jobs to run)

#previw auto focus
#picam2.start(show_preview=True)
#picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
#manual focus (to infiti)
#picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0})

#make one to take full images (capture configuration)
#img_config = picam2.create_still_configuration(lores={"size": (3608, 1592)}, display="lores")
#img_config = picam2.create_still_configuration(lores={"size": (3608, 1592)}, display="lores")#my creation

img_config = picam2.create_still_configuration() #default valeus
picam2.configure(img_config)
picam2.start()
#time.sleep(2)

#image directory
base_directory = "/home/mark/Desktop/imgs"
current_date = time.strftime("%Y-%m-%d")
#"%Y-%m-%d-%H-%M-%S"
# Create the folder for the current date
date_folder = os.path.join(base_directory, current_date)
os.makedirs(date_folder, exist_ok=True)
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
camera_name=picam2.camera_properties['Model']+"_rep1"#get the camera model/info rep1 for pi238

image_filename = f"{timestamp}_{camera_name}.png"
image_path = os.path.join(date_folder, image_filename)

#picam2.capture_file(base_directory+timestamp+"/.png")#store image

#Establish the focus to infity (solve the bluryness of the close objects)
picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0})# Focus to infity

#leave some time to allow for configureations to establish
time.sleep(2)


########################
#check camera properties and configuration details
#picam2.camera_properties
#img_config
#picam2.sensor_modes

############################ capture
picam2.capture_file(image_path) #capture and store image
#Signal end of execution
#image = Image.open(image_path)
#image.show()
print("DONE")
