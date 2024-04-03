# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:00:57 2023

@author: Mark Iradukunda
"""
#import neccessary libraries (assuming they are already installed)
import glob
from matplotlib import pyplot as plt
import numpy as np
import csv
import cv2
import os
from plantcv import plantcv as pcv

#This function locates where plants are and will help to count plants.
def get_roi_circles(line_thickness, coordinates, radius, spacing, nrows, ncols):
    rois = []
    y_start, x_start = coordinates
    for row in range(nrows):
        for col in range(ncols):
            y = y_start + row * spacing
            x = x_start + col * spacing
            rois.append((x, y, radius, line_thickness))
    return rois

# Code to analyze a single image (all images with will be analysed based everything you define in this funtion)
def analyze_single_image(filename, minPxs, calibrator, Histmin, Histmax, rois):
    
    # Treating the string into a list following the separator "."
    fname = filename.rsplit(".", 1)[0]
# Plantcv based colorspaces and gray image
    img, path, filename = pcv.readimage(filename=filename)
    #Draw ROIs on the image
    #for roi in rois:
        #cv2.circle(img, (int(roi[0]), int(roi[1])), int(roi[2]), (0, 255, 0), 2)
    # Do this to transpform the image to original RGB before converting to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.close()
    colorspaces = pcv.visualize.colorspaces(
        rgb_img=img, original_img=False)
    plt.imshow(colorspaces)
    plt.close()
    # pic a color space that shows a better contrast between background and forground and call it gray
    # replace lab with hsv for hsv color pixels
    gray = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    plt.imshow(gray)
    plt.close()
    # show histogram of pixel frequency
    Hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # threshold the grayscaled-image as two channel (black and white) based on threshold values
    Threshold = Histmin + np.argmin(Hist[Histmin:Histmax])
    # Print the histogram plot of 'gray', #Convert the y-axis of histogram in log scale, #Save the histogram plots as png files, #close the file
    plt.plot(Hist)#If you would like to see what is happening behind the seen (best for testing, additional memory required)
    plt.close()

   #The function yields the information of the every seperated components from the thresholded two-channel-image
    ret, thres = cv2.threshold(gray, Threshold, 255, cv2.THRESH_BINARY)
    plt.imshow(thres)
    thres = cv2.bitwise_not(thres)
    plt.close()
   
   #labels: matrix size, stats: the stats in the matrix, centroids: x and y locations within the matrix
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thres) 
    
    # CC_STAT_AREA: function to get area from the stats in the components of the image. it can be changed to width or height of the image; please refer https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
    # with the 'stats' from the two-channel treshold, area can be calculated.
    areas = stats[1:, cv2.CC_STAT_AREA]
    # empty matrix, will be used to write the thresholded two-channel image
    result = np.zeros((labels.shape), np.uint8)

 # Define individual plants as objects
    Objects = 0
    mask = np.zeros_like(thres)

    # Keep track of ROIs containing objects
    rois_with_objects = set()

    # For loop statement to remove pixels outside of the given range 
    #(you want to select objects with a particular size that belongs to plant material)
    for i in range(0, nlabels - 1):
        if areas[i] > minPxs:

            # Convert only the components meeting the conditions
            result[labels == i + 1] = 255
            Objects += 1

            # Check which ROI the current object belongs to
            for idx, roi in enumerate(rois):
                x, y, radius, _ = roi
                if x - radius < centroids[i + 1][0] < x + radius and y - radius < centroids[i + 1][1] < y + radius:
                    rois_with_objects.add(idx)

    # Count objects that partially touch the line (up to 4 objects)
    #Here you know maximum plants in your image. Each ROI is corresponts to a pot or cell in a tray.
    #(You want all objects in that ROI to counted as one object/plant)
    object_count = min(4, len(rois_with_objects))
    #object_count = len(rois_with_objects) would also work if the plant is well formed without touching  

    # Calculate germination percent (you need to know number of seeds you initially planted)
    germination_percent = object_count / 4
   
    # Generate thresholded two channel images along with their original filename
    cv2.imwrite(str(fname)+"_filtered.png", result)  # Save a masked image
    plt.plot(Hist), plt.yscale('log'), plt.savefig(str(fname)+"_histogram.png"), plt.close()


    # Convert the 8-bit image as a format of float 64, that allows to calculaste pixel intensity 
    # CFI_img_f64 = np.float64(gray) #convert 'gray' to float64, which is used to calcuate normalized avg and std
    CFI_img_f64 = np.float64(gray) #For RGB images

    # Average or std of Intensity (range of 0-255); Range for calculation is determined by 'Threshold' value
    Avg_CFI = np.mean(CFI_img_f64[CFI_img_f64>=Threshold]) #Average replace Threshold to threshold
    Std_CFI = np.std(CFI_img_f64[CFI_img_f64>=Threshold]) #standard deviation 
    
    
    pixel_number = cv2.countNonZero(result) #Count the pixel number of white (255) of the two chanel image
    area = pixel_number//(calibrator*calibrator) #Convert the pixel number as an area upon the calibrator (pixel number of a known distance)
    AvgArea = area/(Objects)
    # Experienced ZeroDivisionError when there is no plant yet(error handling 'except' is needed to avoid script stopping)

# Calculate standard deviation:
    SumAreaSquared = 0
    for i in range(0, nlabels - 1):    
       if areas[i] > minPxs:  #if the components within the image that are larger than a specified minimum pixel size, keep and others are discarded
           AreaSquared = (areas[i]-AvgArea)*(areas[i]-AvgArea)
           SumAreaSquared = SumAreaSquared + AreaSquared 
    Std_Area = np.sqrt(SumAreaSquared/(Objects-1))

#if you want to analyse individual plants
    areas_plant = []
    for i in range(0, nlabels - 1):
        if areas[i] > minPxs:
            areas_plant.append(areas[i])

    # Show the image with ROIs to make sure the function is selected the regions where plants are present
    #plt.imshow(img)
    #plt.show()

    #return filename, Threshold, pixel_number, Avg_CFI, Std_CFI, area, Objects, AvgArea, Std_Area to be dropped in .CSV file
    return fname, Threshold, pixel_number, Avg_CFI, Std_CFI, area, object_count, AvgArea, Std_Area, germination_percent


#Function to handle file/folder directories and intial variables (more info below)
def PxEx(minPxs, calibrator, imageformat, Histmin, Histmax, csvname, directory):
    
    # tells program where to look for images. D0 NOT change. If you need to change the folder, do so near the bottom of the program.
    path = directory
    # Make sure to specify the correct file extension for your images!!!
    fileList = glob.glob(path+'/*/*/*'+imageformat) # add slash to specify subfolder, * for wildcard, then imageformat
    # A for loop if statement to extract all file list except 'histogram.png' and 'filterd.png'
    for a in fileList[::-1]:
        # if the list contains a filename matches with these texts
        if a.find('_filtered.png') > -1 or a.find('_filtered.png') > -1 or a.find('histogram.png') > -1:
            fileList.remove(a)  # remove these filenames from the list
        if a.find('_filtered.png') > -1:
            os.remove(a)
            #field names for the .CSV file (you can or remove field names but they have to be defined somewhere )
    field_names = ["File name", "Minimum threshold", "Pixel Number", "CFI_Intensity_avg", "CFI_Intensity_std",
                   "Area", "Object Count", "Avg Area", "Stdev of Area", "Germination Percent"]

    
    with open(csvname, 'w', newline='') as csvfile:  # to create csv file
        writer = csv.DictWriter(csvfile, fieldnames=field_names)  # header of each column within the csv file
        writer.writeheader()
       
        #Get the regions of interest as circles
    rois = get_roi_circles(line_thickness=10, coordinates=(360, 360), radius=250, spacing=600, nrows=2, ncols=2)

    # A For loop statement: iteration from all indices (the filenames) within the folder
    for fdx, filename in enumerate(fileList):
        try:
            # Analyze the current image
            fname, Threshold, pixel_number, Avg_CFI, Std_CFI, area, object_count, AvgArea, Std_Area, germination_percent = analyze_single_image(filename, minPxs, calibrator, Histmin, Histmax, rois)
            #result = analyze_single_image(filename, minPxs, calibrator, Histmin, Histmax)

           #Open and modifie the .CSV file (write)
            with open(csvname, "a", newline='') as csvfile: 
                    writer = csv.writer(csvfile)
                    writer.writerow([os.path.basename(filename), Threshold, pixel_number, Avg_CFI, Std_CFI, area, object_count, AvgArea, Std_Area, germination_percent])
            csvfile.close()
        except Exception as e:
            # Log the error and continue to the next image
            print(f"Error analyzing image {filename}: {e}")#if an image file is trouble, it will move to the next instead of stopping

            # Write default values (zeros or NAs) for the problematic image to the CSV file (to maintain the consistent data dimensions)
            with open(csvname, "a", newline='') as csvfile: 
                writer = csv.writer(csvfile)
                #writer.writerow([filename, 'nan','nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'])
                writer.writerow([os.path.basename(filename), 'nan','nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'])  # Use 'nan' for missing values
            csvfile.close()

'''
# Input code              
# Parameter 1) Set the lower boundary of the pixel size of individual objects in the image that is 
#                    included in the pixel count. This can be used to remove background noise. 
# Parameter 2) This is a calibration factor to convert pixels to area (pixels per mm). If you don't know that value yet (typical) leave this at 1 and convert later in Excel
# Parameter 3) This is the format of your image. You can change png (default now) as any other format.
# Parameter 4 and 5) They are the minimum and maximum values for adjusting threshold. 
#           e.g.: Threshold = 109 + np.argmin(Hist[109:130]); 109 is an example minimum value when the background is dark
#           Users can change the values to find the best values for images (see line 37 for the detail); No change 130 unless your canopy is very bright
# Parameter 6) The name of output file (only csv format allowed) that writes all numbers getting from the program
# Parameter 7) This is where you can specify the folder with your images. The simple solution is to always put your images in the same folder,
#                   analyze them, and then move them to a permamanet folder. That way, you do not need to ever change the program itself.
#              *This program requies '/' for directory, although directory in Window uses '\'. 
'''
#r is used to lender directories readable regardless of device directory formats (PC vs Mac)

PxEx(80, 4.777777778, '.png', 109, 130, r'.\result.csv', r'.\imgs')

#signal end of excecution
print("DONE")