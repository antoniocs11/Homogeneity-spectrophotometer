# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:18:09 2023

@author: usuario
"""
# Libraries needed for the program

import cv2
import os
import datetime as dt
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import tensorflow
import joblib
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pyplot import figure
from tkinter import *
from pathlib import Path
from tkinter import filedialog
from PIL import Image
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy import signal
import csv
from tkinter import messagebox
import tkinter as tk
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


#Necessary code to create the interface through library "Tkinter"

raiz= Tk()
raiz.title("Spectrophotometer")

#Function to extract path to the directory where the current script is located. It will be used later

def directory_path():
    # Gets the absolute path of the current script
    script_path = os.path.abspath(__file__)

    # Gets the directory of the current script
    directory_script = os.path.dirname(script_path)

    # Modifies the directory path so that it can be read by python at a later time
    directory_process = directory_script.replace("\\" , "\\\\")

    return directory_process

#----------------------------Spectrophotometer Functions-------------------------------------------


# The variables named with "global" are to be stored outside this function and can be used in any area of the code and program.
"""
#Function "examine" is associated with button 5, which is responsible for loading the file to be studied. 
This function stores the file path and displays it on the interface. It also resets all the text boxes of the interface 
and the graph that appears on the right.

"""
def examine():
    global name1 #variable to save path of  selected video or image 
    global example1 #variable to show this path in the screen of the interface in the text box 3.
    name1 = filedialog.askopenfilename()
    example1.set(name1)
    
    #All text boxes of the interface are shown in blank, except text box 3 which is showing selected path. 
    wave.set("") #variable associated with text box 10
    wave1.set("") #variable associated with text box 5
    tframes.set("") #variable associated with text box 9
    tframes1.set("") #variable associated with text box 4
    process.set("") #variable associated with text box 6
    saved.set("") #variable associated with text box 8
    
    
"""
#Function "timef" is associated with button 16, which is responsible to saved time, introduce for the user (in the text box 4),
in the system and be able to show in the text box 9
"""
def timef():
    global time #variable that save the time introduce for the user in the text box 4
    time=float(text_box4.get()) #The time added by the user is saved in the variable "time"
    tframes.set(time) #The time introduced by the user appears in the text box 9
    time=float(text_box9.get())
    tframes1.set("") #Text box 4 is changed to blank

"""
#Function "landa" is associated with button 17, which is responsible to save wavelength, introduce by the user(in the text box5),
in the system and be able to show it in the text box 10
"""
def landa():
    global lamda #variable that save the wavelength introduce by the user in the text box 5
    lamda=int(text_box5.get())#The wavelength add by the user is saved in the variable "lamda"
    wave.set(lamda)#The wavelength introduced by the user appears in the text box 10
    lamda=int(text_box10.get())
    wave1.set("") #Text box 5 is changed to blank

"""
#Function "measure" allows the user to load an image. The image can be either the first frame of a selected video or an individual 
image file. Once the image is opened, the user can interactively select a portion of the image that corresponds to a physical 
reference of 1 cm. This selection is crucial to establish a scaling factor, enabling the program to relate pixel distances in 
the image to real-world measurements accurately.
"""

def measure():
    
    
    global d_xaxis #variable to store the x-axis distance of the area previously selected by the user.
    global d_yaxis #variable to store the y-axis distance of the area previously selected by the user.
    global pixel_1cm #variable to store the number of pixels 1 cm is in the image
    global m_1cm #variable to store cm/pixel
    
    #function to open an image and select an area
    
    def drawing1(event,x,y,flags,param):
        global pixelx1_1, pixelx2_2, pixely1_1, pixely2_2   #variables that will save pixels of the selected area by the user
        
        
        if event == cv2.EVENT_LBUTTONDOWN: # By clicking the left mouse button, the upper left corner of the area to be studied is selected.
            
            #Pixels in X axis and Y axis are saved in variable "pixelx1" and "pixely1"
            pixelx1_1=x
            pixely1_1=y
            #The selected pixels are displayed on the screen
            print ('pixel x1=',x) 
            print ('pixel y1=',y)
        if event == cv2.EVENT_RBUTTONDOWN: # By clicking the right mouse button, the bottom right corner of the area to be studied is selected.
            
            #Pixels in X axis and Y axis are saved in variable "pixelx2" and "pixely2"
            pixelx2_2=x
            pixely2_2=y
            #The selected pixels are displayed on the screen
            print ('pixel x2=',x)
            print ('pixel y2=',y)
            cv2.rectangle(image,(pixelx1_1,pixely1_1),(pixelx2_2,pixely2_2),(255,0,0),1)
    
    if option1==1: #User has selected "Video" option
        
        image = cv2.imread('./data/frame0.jpg') #First frame of the selected video is opened
        image_to_study = image.copy() 
    
        cv2.namedWindow('image_to_study')
        cv2.setMouseCallback('image_to_study',drawing1) # "drawing1" function is called 
    
    
        while True:
        	cv2.imshow('image_to_study',image)
        	
        	if cv2.waitKey(1) & 0xFF == 27:
        		break
    
        cv2.destroyAllWindows()
        
        #the user has selected the x-axis distance of 1 cm in the image. The number of pixels that 1 cm contains in the image and the size of each pixel is then calculated. 
        
        pixel_1cm=abs(pixelx2_2-pixelx1_1) #pixel/cm
        m_1cm=10/pixel_1cm #cm/pixel
        
        #The X- and Y-axis distance of the area selected by the user in the previous step is then determined. 
        
        d_xaxis=x_axis*m_1cm #the x-axis measure of the area previously selected by the user.
        d_yaxis=y_axis*m_1cm  #the y-axis measure of the area previously selected by the user.
    
    if option1==2: #User has selected "Image_sequence" option
        
        image = cv2.imread('./'+str(name6)) ##First image of the selected folder is opened
        image_to_study = image.copy() 
    
        cv2.namedWindow('image_to_study')
        cv2.setMouseCallback('image_to_study',drawing1) # "drawing1" function is called
        
        while True:
        	cv2.imshow('image_to_study',image)
        	
        	if cv2.waitKey(1) & 0xFF == 27:
        		break
    
        cv2.destroyAllWindows()
        
        pixel_1cm=abs(pixelx2_2-pixelx1_1) #pixel/cm
        m_1cm=10/pixel_1cm #cm/pixel
        
        d_xaxis=x_axis*m_1cm #the x-axis measure of the area previously selected by the user.
        d_yaxis=y_axis*m_1cm #the y-axis measure of the area previously selected by the user.
    
    button4.config(bg="orange") #the colour of the button changes to indicate that this function has already been carried out.
        
"""
#Function "calculate" is associated with button 8, which is responsible to show the first frame of the video or the first image of 
the folder selected in another window. In this window, the user could select the desired area to study.

"""

def calculate():
    
    global n #variable that store the number of obtained frames of the video selected or the number of images that appeared in the folder selected by the user. 
    global nx # number of different wavelength in each spectrum 
    global sol_n_n # variable to store neural network 
    global Labim3 #variable to store Lab coordinates  
    global Labim_norm4 #variable to store normalised Lab coordinates
    global dt #variable to store number of divisions of the selected area 
    global dx #variable to store parts into which the area selected by the user on the x-axis is divided.
    global dy #variable to store parts into which the area selected by the user on the y-axis is divided.
    global x_axis # variable to store number of pixels occupied by a row (x-axis) of the selected area
    global y_axis # variable to store number of pixels occupied by a column (y-axis) of the selected area
           
    nx=61 #The number of different wavelengths in each spectrum will be 61, one per 5 nm (400-700 nm)
    
    if option1==1: #User has selected "Video" option
        
        # Read the video from specified path saved in variable "name1"
        
        vid = cv2.VideoCapture(name1)
        
        #A new folder is created (in the same folder where the script is saved) to save all frames that will be extracted from the video
    
        try:
    
            # creating a folder named data
            if not os.path.exists('data'):
                os.makedirs('data')
    
        # if not created then raise error
        except OSError:
            print('Error: Creating directory of data')
    
        # Two variables are set that will serve as counters to store all the frames extracted from the video in the subsequent loop.
        currentframe = 0
        frame_count = 0
        fps = vid.get(cv2.CAP_PROP_FPS) #variable to extract the number of frames per second of the video
        interval_frames = int(fps * time) #number of frames specifying how often a frame should be extracted
        n=0  #reset of the variable "n"
        while (True):
    
            # reading from frame
            success, frame = vid.read()
                        
            if success:
                                
                # continue creating images until video remains
                if frame_count % interval_frames == 0:
                    
                    n=n+1
                    name = './data/frame' + str(currentframe) + '.jpg'
                    print('Creating...' + name)
                    
    
                    # writing the extracted images
                    cv2.imwrite(name, frame)
    
                    # increasing counter so that it will show how many frames are created
                    currentframe += 1
                    
            else:
                
                
                break
            frame_count += 1
    
        # Release all space and windows once done
        vid.release()
        cv2.destroyAllWindows()
    
        #SELECTION OF THE PART OF THE IMAGE TO BE EVALUATED
            
        #Function for the selection of the area of video extracted frames to process
        def drawing(event,x,y,flags,param):
            global pixelx1, pixelx2, pixely1, pixely2   #variables that will save pixels of the selected area by the user
            
            
            if event == cv2.EVENT_LBUTTONDOWN: # By clicking the left mouse button, the upper left corner of the area to be studied is selected.
                
                #Pixels in X axis and Y axis are saved in variable "pixelx1" and "pixely1"
                pixelx1=x
                pixely1=y
                #The selected pixels are displayed on the screen
                print ('pixel x1=',x) 
                print ('pixel y1=',y)
            if event == cv2.EVENT_RBUTTONDOWN: # By clicking the right mouse button, the bottom right corner of the area to be studied is selected.
                
                #Pixels in X axis and Y axis are saved in variable "pixelx2" and "pixely2"
                pixelx2=x
                pixely2=y
                #The selected pixels are displayed on the screen
                print ('pixel x2=',x)
                print ('pixel y2=',y)
                cv2.rectangle(image,(pixelx1,pixely1),(pixelx2,pixely2),(255,0,0),1)
        	
        #Code to show the first frame obtained of the video in another window and code necessary to call the previous function
        image = cv2.imread('./data/frame0.jpg')
        image_to_study = image.copy() 
    
        cv2.namedWindow('image_to_study')
        cv2.setMouseCallback('image_to_study',drawing)
    
    
        while True:
        	cv2.imshow('image_to_study',image)
        	
        	if cv2.waitKey(1) & 0xFF == 27:
        		break
    
        cv2.destroyAllWindows()
        
        x_axis=abs(pixelx2-pixelx1) #Variable to store the number of pixels in x axis from the area selected by the user
        y_axis=abs(pixely2-pixely1) #Variable to store the number of pixels in y axis from the area selected by the user
        #It is shown the numbers of pixels in Y and X axis from the area selected by the user
        print ('x_axis=',x_axis)
        print ('y_axis=',y_axis)
        
        p=x_axis*y_axis #variable to save the number of total pixels from the area selected by the user
        
        dx=dx2
        dy=dy2
        dt=dx*dy 
        pixels_x=int(x_axis/dx) #number of pixels in X-axis in each division of the selected area 
        pixels_y=int(y_axis/dy) #number of pixels in Y-axis in each division of the selected area
        Labim3=np.zeros((n,dt,3))
        
        
        #The function "directory_path" is called to extract directory path in which the script has been saved
        if __name__ == "__main__":
            
            Process_path = directory_path()
        
        #Necessary loop to extract Lab coordinates from each pixel of the selected area. Then, average value of each coordinate is obtained
        for Counter in range(n): 
            image1=Image.open(Process_path + '\\data\\frame' + str(Counter) + '.jpg') #frame by frame is studied, focusing in the area selected by the user
            
           
            pixelx11=pixelx1
            pixely11=pixely1
            c=0
            L=0
            a=0
            bb=0
            #loop that traverses the selected area for each frame, entering in the variable Lab3im the average Lab coordinates of each division, generating Lab dt coordinates in each frame.
            for s in range(dy):
                for z in range(dx):
                    for i in range(pixels_y):
                        for j in range(pixels_x):
                            r, g, b = image1.getpixel((j+pixelx11,i+pixely11))
                            im1 = sRGBColor(r/255, g/255, b/255)
                            im2= convert_color(im1, LabColor) 
                            L=im2.lab_l + 5 + L
                            a=im2.lab_a + a
                            bb=im2.lab_b + bb
                    Labim3[Counter,c,0]=L/(pixels_x*pixels_y)
                    Labim3[Counter,c,1]=a/(pixels_x*pixels_y)
                    Labim3[Counter,c,2]=bb/(pixels_x*pixels_y)
                    pixelx11=pixelx11+pixels_x
                    L=0
                    a=0
                    bb=0
                    c=c+1
                pixelx11=pixelx11-dx*pixels_x
                pixely11=pixely11+pixels_y
            c=0
        
       
        #DATA NORMALIZATION
        
        Labim_norm3=np.zeros((n,dt*nx,4))
        Labim_norm4=np.zeros((n,dt*nx,4))
        
        #Loop to normalise the previously extracted Lab coordinates for subsequent input into the neural network in the correct format.
        
        for z in range(n):
            y=0
            for i in range (dt):
              for j in range (nx):
                Labim_norm3[z,y,0]=Labim3[z,i,0]
                Labim_norm3[z,y,1]=Labim3[z,i,1]
                Labim_norm3[z,y,2]=Labim3[z,i,2]
                Labim_norm3[z,y,3]=400+j*5
                Labim_norm4[z,y,0]=Labim_norm3[z,y,0]/100
                Labim_norm4[z,y,1]=(Labim_norm3[z,y,1]+100)/200
                Labim_norm4[z,y,2]=(Labim_norm3[z,y,2]+100)/200
                
                if Labim_norm3[z,y,3] == 400:
                  Labim_norm4[z,y,3]=0
                else:
                  Labim_norm4[z,y,3]=(Labim_norm3[z,y,3]-400)/300
        
                y=y+1
        
        
        sol_n_n = load_model('Neural_network.h5') # the file containing all the neural network data is loaded. This file should be saved in the same folder as this script
    
        process.set("Process completed") #The process has been finished and the text box 6 shows it
    
    if option1==2: #User has selected "Image sequence" option
        
        global name6       
        #code lines to extract folder name in which the images are
        f=0
        variable=-1
        
        while (f<2):
            
            if name1[variable]=='/':
                
                variable=variable+1
                name6=name1[variable:]
                f=f+1
                variable=variable-2
            else:
                variable=variable-1
        
        f=0
        variable=1
        while (f==0):
            
            if name6[variable]=='/':
                
                name7=name6[:variable]
                f=1
            else:
                variable=variable+1
        
        def drawing(event,x,y,flags,param): #The same function created in video section to obtain the selected area by the user
            global pixelx1, pixelx2, pixely1, pixely2
        	   
            if event == cv2.EVENT_LBUTTONDOWN:
                pixelx1=x
                pixely1=y
                print ('pixel x1=',x)
                print ('pixel y1=',y)
            if event == cv2.EVENT_RBUTTONDOWN:
                pixelx2=x
                pixely2=y
                print ('pixel x2=',x)
                print ('pixel y2=',y)
                cv2.rectangle(image,(pixelx1,pixely1),(pixelx2,pixely2),(255,0,0),1)
        	
        #code to show the selected image by the user in another window and code necessary to call "drawing" function
        image = cv2.imread('./'+str(name6))
        image_to_study = image.copy() 
    
        cv2.namedWindow('image_to_study')
        cv2.setMouseCallback('image_to_study',drawing)
        
        while True:
        	cv2.imshow('image_to_study',image)
        	
        	if cv2.waitKey(1) & 0xFF == 27:
        		break
    
        cv2.destroyAllWindows()
        
        #code lines to extract the images folder path in variable "name8"
        
        f=0
        variable=-1
        
        while (f==0):
            
            if name1[variable]=='/':
                variable=variable+1
                name8=name1[:variable]
                f=1
            else:
                variable=variable-1
        
        folder = name8
    
        # Gets the list of files in the folder
        files = os.listdir(folder)
    
        # Filter the list to include only files (not directories).
        files = [file for file in files if os.path.isfile(os.path.join(folder, file))]
    
        # Count the number of files, number of images in the selected folder
        n = len(files)
        
        x_axis=abs(pixelx2-pixelx1) #variable to store the number of pixels in x axis from the area selected by the user
        y_axis=abs(pixely2-pixely1)#variable to store the number of pixels in y axis from the area selected by the user
        #The numbers of pixels in Y and X axis corresponding to the area selected by the user are shown
        print ('x_axis=',x_axis)
        print ('y_axis=',y_axis)
    
        # Extract images and convert into color coordinates
        
        p=x_axis*y_axis #variable to save the number of total pixels from the selected area by the user
        
        dx=dx2
        dy=dy2
        dt=dx*dy
        pixels_x=int(x_axis/dx)
        pixels_y=int(y_axis/dy)
        Labim3=np.zeros((n,dt,3))
        
        counter=0
        file=0
        global File_path2
        for file in os.listdir(folder): #Necessary loop to open each image and extract Lab coordinates from the selected area 
            # Complete file path
            File_path2 = os.path.join(folder, file)
        
            # Opens each image in the selected folder
            if os.path.isfile(File_path2) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image1=Image.open(File_path2)
                       
            
            #Necessary loop to obtain Lab coordinates from each pixel in each image
            pixelx11=pixelx1
            pixely11=pixely1
            c=0
            L=0
            a=0
            bb=0
            #loop that traverses the selected area for each image, entering in the variable Lab3im the average Lab coordinates of each division, generating Lab dt coordinates in each image.
            for s in range(dy):
                for z in range(dx):
                    for i in range(pixels_y):
                        for j in range(pixels_x):
                            r, g, b = image1.getpixel((j+pixelx11,i+pixely11))
                            im1 = sRGBColor(r/255, g/255, b/255)
                            im2= convert_color(im1, LabColor) 
                            L=im2.lab_l + L
                            a=im2.lab_a + a
                            bb=im2.lab_b + bb
                    Labim3[counter,c,0]=L/(pixels_x*pixels_y)
                    Labim3[counter,c,1]=a/(pixels_x*pixels_y)
                    Labim3[counter,c,2]=bb/(pixels_x*pixels_y)
                    pixelx11=pixelx11+pixels_x
                    L=0
                    a=0
                    bb=0
                    c=c+1
                pixelx11=pixelx11-dx*pixels_x
                pixely11=pixely11+pixels_y
                
            c=0
            
            counter=counter+1
                
        #DATA NORMALIZATION
        
        Labim_norm3=np.zeros((n,dt*nx,4))
        Labim_norm4=np.zeros((n,dt*nx,4))
        
        #Loop to normalise the previously extracted Lab coordinates for subsequent input into the neural network in the correct format.
        
        for z in range(n):
            y=0
            for i in range (dt):
              for j in range (nx):
                Labim_norm3[z,y,0]=Labim3[z,i,0]
                Labim_norm3[z,y,1]=Labim3[z,i,1]
                Labim_norm3[z,y,2]=Labim3[z,i,2]
                Labim_norm3[z,y,3]=400+j*5
                Labim_norm4[z,y,0]=Labim_norm3[z,y,0]/100
                Labim_norm4[z,y,1]=(Labim_norm3[z,y,1]+100)/200
                Labim_norm4[z,y,2]=(Labim_norm3[z,y,2]+100)/200
                
                if Labim_norm3[z,y,3] == 400:
                  Labim_norm4[z,y,3]=0
                else:
                  Labim_norm4[z,y,3]=(Labim_norm3[z,y,3]-400)/300
        
                y=y+1
              
        
        sol_n_n = load_model('Neural_network.h5') #the file containing all the neural network data is loaded. This file should be saved in the same folder as this script
        
        process.set("Process completed") #The process has been finished and the text box 6 shows it


def T_study():
    
     
    global wlength #variable to create the entire wavelength range between 400 and 700 nm
    global s #variable to know the method selected by the user. This variable is related to the "PDF" function
    global s_pred #variable to store Transmittance spectra from the neural network. This variable is not neccesary to be global, but it can be useful to test in variable explorer
    global s_pred_reshaped # s_pred variable in other format to work later.This variable is not neccesary to be global, but it can be useful to test in variable explorer
    global transmittance46 # variable to store maximum contrast in eachh part of the selected area of the images or frames
    global lamda2 # variable to store user-entered standarized wavelength 
    global espectros #variable to store Transmittance spectra from neural network in other format. This variable will called later
    global transmittance46_matrix # variable to store values of "contrast46", but in other format that let us to use later
    global Labim_color
    global transmittance1
    global ae
    global ae_matrix
    global w
    s=4
    
    wlength=np.zeros((61,1))
    z=0
    for i in range (61):
      wlength[i] = 400 + z
      z=z+5
    
    #loop to store transmittance spectra from the previously loaded neural network
    s_pred=np.zeros((nx*dt,n))
    for i in range(n):
        
        # Prediction is carried out
        prediccion = sol_n_n.predict(Labim_norm4[i,:,:])
        # Ensures that the prediction has the correct form (nx*dt,)
        prediccion = prediccion.flatten()
        # Stores the prediction in the corresponding column of s_pred
        s_pred[:, i] = prediccion
    
    assert s_pred.shape == (nx*dt, n), "s_pred does not have the expected dimensions"
    s_pred_reshaped = s_pred.T.reshape(n, dt, nx)
    
    transmittance46=np.zeros(dt)
    
    
    # the wavelength entered by the user is stored as a normalised wavelength
    
    if lamda==400:
        lamda2=0
    else:
        lamda2 = int((lamda-400)/5) #user-entered standarized wavelength 
    
    for i in range(dt):
        
        transmittance46[i]=s_pred_reshaped[0,i,lamda2]*100
            
    assert len(transmittance46) == dx * dy, "Transmittance46 not the right size"
    transmittance46_matrix = np.array(transmittance46).reshape((dx, dy))
    
    espectros=np.zeros((dt,nx))
    
    for i in range (dt):
        
        for j in range (nx):
            espectros[i,j]=s_pred_reshaped[0,i,j]
    
    #colorimetry study
    
    dw=dx*dy
    Labim_color=np.zeros((1,3))
    Labim_color1=np.zeros((dw,3))
    ae=np.zeros(dw)
    
    Labim_color[0,0]=Labim3[0,0,0]
    Labim_color[0,1]=Labim3[0,0,1]
    Labim_color[0,2]=Labim3[0,0,2]
    transmittance1=abs(transmittance46[0]-dz2)
    
    for i in range(dw):
        
        if abs((transmittance46[i])-dz2)<transmittance1:
            
            Labim_color[0,0]=Labim3[0,i,0]
            Labim_color[0,1]=Labim3[0,i,1]
            Labim_color[0,2]=Labim3[0,i,2]
            transmittance1=abs(transmittance46[i]-dz2)
            w=i
    
    for i in range(dw):
        
        ae[i]=((Labim3[0,i,0]-Labim3[0,w,0])**2+(Labim3[0,i,1]-Labim3[0,w,1])**2+(Labim3[0,i,2]-Labim3[0,w,2])**2)**0.5
        
    
    assert len(ae) == dx * dy, "ae not the right size"
    ae_matrix = np.array(ae).reshape((dx, dy))
        
        
    
    #Code lines to graphically represent spectra results from the neural network in the interface.
    
    fig, axs =plt.subplots(1,1,dpi=100, figsize=(7,5), sharey=True)
    fig.suptitle('Transmittance Homogeneity', size=20)
    
    x_ticks = np.linspace(-0.5, dy - 0.5, dy+1)
    y_ticks = np.linspace(-0.5, dx - 0.5, dx+1)
    #x_labels = np.linspace(0, d_xaxis, dy)
    #y_labels = np.linspace(0, d_yaxis, dx)
    x_labels = np.append(np.linspace(0, d_xaxis - (d_xaxis / dy), dy), d_xaxis)  
    y_labels = np.append(np.linspace(0, d_yaxis - (d_yaxis / dx), dx), d_yaxis)
    
    y_labels = y_labels[::-1]
    # Define the colour scale and boundaries
    #cmap = plt.cm.viridis
    #cmap=cm.get_cmap("Oranges_r")
    cmap=mcolors.LinearSegmentedColormap.from_list("black", ["black", "white"])
    vmin = min(transmittance46)  #  Minimum value of contrast46
    vmax = max(transmittance46)  #  Maximum value of contrast46
    
    
    #Colour map created from contrast46_matrix
    
    plt.imshow(transmittance46_matrix, cmap=cmap, aspect='auto', interpolation='bilinear', norm=Normalize(vmin=vmin, vmax=vmax))
    
    plt.xticks(ticks=x_ticks, labels=[f'{label:.1f} mm' for label in x_labels])
    plt.yticks(ticks=y_ticks, labels=[f'{label:.1f} mm' for label in y_labels])
    plt.colorbar(label='Transmittance (%)')
    
    
    canvas=FigureCanvasTkAgg(fig, master=frame2)
    canvas.draw()
    canvas.get_tk_widget().grid(column=4, row=1, rowspan=10)
    
    button18.config(bg="orange") #the colour of the button changes to indicate that this function has already been carried out.
            
    
"""        
#Function "CP" is associated with button 9. It will be executed when the user decide to select "Contrast study" process. 
This function generates the transmittance spectra of each zone of the images or frames selected by the user. 
From them, it extracts the contrasts in the wavelength exposed by the user and displays a colour map of these contrasts 
on the interface.

"""

def CP():
    
    global contrast2 #variable to store maximum contrast from the obtained spectra 
    global wlength #variable to create the entire wavelength range between 400 and 700 nm
    global s #variable to know the method selected by the user. This variable is related to the "PDF" function
    global s_pred #variable to store Transmittance spectra from the neural network. This variable is not neccesary to be global, but it can be useful to test in variable explorer
    global s_pred_reshaped # s_pred variable in other format to work later.This variable is not neccesary to be global, but it can be useful to test in variable explorer
    global contrast46 # variable to store maximum contrast in eachh part of the selected area of the images or frames
    global lamda2 # variable to store user-entered standarized wavelength 
    global espectros #variable to store Transmittance spectra from neural network in other format. This variable will called later
    global contrast46_matrix # variable to store values of "contrast46", but in other format that let us to use later
    
    s=1 #It is associated with number 1 in "PDF" function
    
    #loop to create an array to store the wavelengths from 400 to 700, every 5 nm. This array use to graph Transmittance spectra later
    
    wlength=np.zeros((61,1))
    z=0
    for i in range (61):
      wlength[i] = 400 + z
      z=z+5
    
    #loop to store transmittance spectra from the previously loaded neural network
    s_pred=np.zeros((nx*dt,n))
    for i in range(n):
        
        # Prediction is carried out
        prediccion = sol_n_n.predict(Labim_norm4[i,:,:])
        # Ensures that the prediction has the correct form (nx*dt,)
        prediccion = prediccion.flatten()
        # Stores the prediction in the corresponding column of s_pred
        s_pred[:, i] = prediccion
    
    assert s_pred.shape == (nx*dt, n), "s_pred does not have the expected dimensions"
    s_pred_reshaped = s_pred.T.reshape(n, dt, nx)
    
    
    contrast46=np.zeros(dt)
    contrast2=np.zeros((dt,2))
    
    # the wavelength entered by the user is stored as a normalised wavelength
    
    if lamda==400:
        lamda2=0
    else:
        lamda2 = int((lamda-400)/5) #user-entered standarized wavelength 
    
    #loop to store in the array contrast2 the maximum and minimum transmittance values of each part of the image or frame (in the wavelength entered by the user), to store later in the array contrast46 the contrast. 
    
    for i in range(dt):
        max_v=0
        min_v=1
        for j in range(n):
            
            value=s_pred_reshaped[j,i,lamda2]
            if value>max_v:
                max_v=value
                contrast2[i,0]=j
            if value<min_v:
                min_v=value
                contrast2[i,1]=j
        
        contrast46[i]=(max_v-min_v)*100
    
    assert len(contrast46) == dx * dy, "contrast46 not the right size"
    contrast46_matrix = np.array(contrast46).reshape((dx, dy))
                   
    espectros=np.zeros((2*dt,nx))
    
    #loop to store the contrast values in another format that allows them to be plotted on a colour map.
    
    for i in range (dt):
        
        x1=int(contrast2[i,0])
        x2=int(contrast2[i,1])
        for j in range (nx):
            espectros[2*i,j]=s_pred_reshaped[x1,i,j]
            espectros[2*i+1,j]=s_pred_reshaped[x2,i,j]

    
    #Code lines to graphically represent spectra results from the neural network in the interface.
    
    fig, axs =plt.subplots(1,1,dpi=100, figsize=(7,5), sharey=True)
    fig.suptitle('Contrast', size=20)
    
    x_ticks = np.linspace(-0.5, dy - 0.5, dy+1)
    y_ticks = np.linspace(-0.5, dx - 0.5, dx+1)
    #x_labels = np.linspace(0, d_xaxis, dy)
    #y_labels = np.linspace(0, d_yaxis, dx)
    x_labels = np.append(np.linspace(0, d_xaxis - (d_xaxis / dy), dy), d_xaxis)  
    y_labels = np.append(np.linspace(0, d_yaxis - (d_yaxis / dx), dx), d_yaxis)
    
    y_labels = y_labels[::-1]
    # Define the colour scale and boundaries
    #cmap = plt.cm.viridis
    #cmap=cm.get_cmap("Oranges")
    cmap=mcolors.LinearSegmentedColormap.from_list("black", ["white", "black"])
    vmin = min(contrast46)  #  Minimum value of contrast46
    vmax = max(contrast46)  #  Maximum value of contrast46
    
    
    #Colour map created from contrast46_matrix
    
    plt.imshow(contrast46_matrix, cmap=cmap, aspect='auto', interpolation='bilinear', norm=Normalize(vmin=vmin, vmax=vmax))
    
    plt.xticks(ticks=x_ticks, labels=[f'{label:.1f} mm' for label in x_labels])
    plt.yticks(ticks=y_ticks, labels=[f'{label:.1f} mm' for label in y_labels])
    plt.colorbar(label='Contrast (%)')
    
    
    canvas=FigureCanvasTkAgg(fig, master=frame2)
    canvas.draw()
    canvas.get_tk_widget().grid(column=4, row=1, rowspan=10)
    
    button9.config(bg="orange") #the colour of the button changes to indicate that this function has already been carried out.

"""
#Function associated with button 10. It will be executed when the user decide to select "Stability process" process.
This function calculates the transmittance values at the user-specified length of each frame or image. 
It calculates this transmittance evolution over time and then calculates the upper and lower envelopes of the resulting function. 
Then, by subtracting these envelopes, a contrast evolution curve is obtained from which the characteristic stability parameter N80 
is obtained.

"""
def SP():
     
    
    global s_pred #variable to store Transmittance spectra from the neural network. This variable is not neccesary to be global, but it can be useful to test in variable explorer
    global s_pred_reshaped # s_pred variable in other format to work later.This variable is not neccesary to be global, but it can be useful to test in variable explorer
    global N80 #variable to store the parameter N80 of each division of the selected area 
    global data_for_nx #Variable to store transmittance evolution of the images or frames 
    global saved_data #variable to store important parameters to use later in "PDF" function
    global N80_matrix #N80 variable but in other format to use in a colormap
    global diff_curves_array #variable to store evolution of contrast to use later in "PDF" function
    global s #variable to know the method selected by the user. This variable is related to the "PDF" function
    
    s=3 #It is associated with number 3 in "PDF" function
    
    N80=np.zeros((dt,1))
    
    # variable "x" store number of cycles. Each frame or image represent half of a cycle. In this way, "x" will be X-axis to graph some results
    x=np.zeros((n,1))
    x[0]=0
    for i in range(n-1):
        x[i+1]=x[i]+0.5
    
    s_pred=np.zeros((dt,n))
    
    #lamda3 store normalised wavelength introduced for the user 
    if lamda==400:
        lamda3=0
    else:
        lamda3=(lamda-400)/300
        
        
    #loops to, firstly, isolate the Lab coordinates normalised to the wavelength entered by the user and 
    #,secondly, make the transmittance predictions from these Lab coordinates. 
    #In the variable data_for_nx is stored this evolution in the correct format.
    
    Labim_norm5=np.zeros((n,dt,4))

    for i in range(n):
        z=0
        for j in range(dt*nx):
            if Labim_norm4[i,j,3]==lamda3:
                Labim_norm5[i,z,0]=Labim_norm4[i,j,0]
                Labim_norm5[i,z,1]=Labim_norm4[i,j,1]
                Labim_norm5[i,z,2]=Labim_norm4[i,j,2]
                Labim_norm5[i,z,3]=Labim_norm4[i,j,3]
                z=z+1

    for i in range(n):
        
        prediccion = sol_n_n.predict(Labim_norm5[i,:,:])
        prediccion = prediccion.flatten()
        s_pred[:,i] = prediccion*100
        

    assert s_pred.shape == (dt, n), "s_pred does not have the expected dimensions"
    s_pred_reshaped = s_pred.T.reshape(n, dt)

    data_for_nx = s_pred_reshaped
    
    #the form of the function that the evolution of the contrast will have is defined to use later
    def fitting_function(x, A1, t1, y0):
        return A1 * np.exp(-x / t1) + y0
    
    #"saved_data" variable store several important parameters to use later
    saved_data = {
    "y_values": [],
    "envelope_upper": [],
    "envelope_lower": [],
    "diff_curves": [],
    "fit_params": [],
    "x_data": [],
    }
    
    fit_params = []
    #some characteristics of the graphs to be created are initialised
    fig_envelopes, axs_envelopes = plt.subplots(dt, 1, dpi=100, figsize=(8, dt * 2), sharex=True)
    fig_fits, axs_fits = plt.subplots(dt, 1, dpi=100, figsize=(8, dt * 2), sharex=True)
    
    diff_curves_array = np.full((n, dt), np.nan)  # Use np.nan to handle curves of different lengths
    
    #Loop in which the envelopes of the transmittance evolution of the images or frames are created. 
    #Subsequently, the contrast curve (the subtraction of both) is generated and both curves are plotted but in "plots" window.
    
    for i in range(dt):
        y = data_for_nx[:, i] #variable "y" store transmittance evolution 
        
        # Finding peaks (local maxima)
        peaks, _ = find_peaks(y)
        # Find the valleys (local minima) by inverting the signal.
        valleys, _ = find_peaks(-y)
        
        # Obtain values of peaks and valleys
        max_values = y[peaks] if len(peaks) > 0 else [np.nan]  # Avoid errors if there are no peaks
        min_values = y[valleys] if len(valleys) > 0 else [np.nan]  # Avoid errors if there are no valleys
        
        # Get the maximum and minimum of peaks and valleys.
   
        envelope_upper = np.interp(np.arange(n), peaks, max_values, left=np.nan, right=np.nan)
        envelope_lower = np.interp(np.arange(n), valleys, min_values, left=np.nan, right=np.nan)
        
        diff_curve = envelope_upper - envelope_lower # Contrast curve of the test
        
        x_data = np.arange(0, n * 0.5, 0.5) #same information than "x" variable
        
        #The code ensures that x_data and diff_curve contain only valid numeric values.
        
        valid_indices = ~np.isnan(diff_curve) & ~np.isinf(diff_curve)
        x_data = x_data[valid_indices]
        diff_curve = diff_curve[valid_indices]
        
        diff_curves_array[:len(diff_curve), i] = diff_curve
        
        #Some important information is store in "saved _data"
        
        saved_data["y_values"].append(y)
        saved_data["envelope_upper"].append(envelope_upper)
        saved_data["envelope_lower"].append(envelope_lower)
        saved_data["diff_curves"].append(diff_curve)
        saved_data["x_data"].append(x_data)

        #code to ensure that some variables have the correct information and do not contain any anomalous data or incorrect form.        

        if len(diff_curve) < 2:
            fit_params.append([np.nan, np.nan, np.nan])
            saved_data["fit_params"].append([np.nan, np.nan, np.nan])
            
            continue
        
        
        initial_guess = [np.max(diff_curve), 1, np.min(diff_curve)]
        bounds = (0, [np.inf, np.inf, np.inf])  # Adding limits to parameters
        
        try:
            popt, _ = curve_fit(fitting_function, x_data, diff_curve, p0=initial_guess, bounds=bounds)
            fit_params.append(popt)
            saved_data["fit_params"].append(popt)
            
            fitted_curve = fitting_function(x_data, *popt)
        except RuntimeError:
            fit_params.append([np.nan, np.nan, np.nan])
            saved_data["fit_params"].append([np.nan, np.nan, np.nan])


#The evolution of the transmittance over the cycles together with the envelopes created is plotted in "plot" window
        
        axs_envelopes[i].plot(np.arange(n) * 0.5, y, label="Transmittance evolution", color="blue")
        axs_envelopes[i].plot(np.arange(n) * 0.5, envelope_upper, label="Upper envelope", color="green")
        axs_envelopes[i].plot(np.arange(n) * 0.5, envelope_lower, label="Lower envelope", color="red")
        axs_envelopes[i].fill_between(np.arange(n) * 0.5, envelope_lower, envelope_upper, color="yellow", alpha=0.3, label="area between envelopes")
        axs_envelopes[i].legend()
        axs_envelopes[i].set_title(f"Transmittance evolution for area={i+1}")
        axs_envelopes[i].set_xlabel("Number of cycles")
        axs_envelopes[i].set_ylabel("Transmittance (%)")
        
        params = fit_params[i]
#The evolution of the contrast over the cycles together with the fit function is plotted in "plot" window
       
        fitted_curve = fitting_function(x_data, *params)
        axs_fits[i].plot(x_data, diff_curve, label="Differential curve", color="blue")
        axs_fits[i].plot(x_data, fitted_curve, label="Fitted curve", color="orange", linestyle="--")
        axs_fits[i].legend()
        axs_fits[i].set_title(f"Fitted curve of area={i+1}")
        axs_fits[i].set_xlabel("Number of cycles")
        axs_fits[i].set_ylabel("Contrast (%)")


#calculation of parameter N80 and storage in the correct format in N80_matrix 
    
    for i, params in enumerate(fit_params):
        A1, t1, y0 = params
        xx = -t1 * np.log((0.8 * (y0 + A1) - y0) / A1)
        N80[i] = xx
    for i, (params, xx) in enumerate(zip(fit_params, N80)):
        print(f'Fitting parameters of area={i+1}: A1={params[0]}, t1={params[1]}, y0={params[2]}')
        print(f'N80 parameter of area={i+1}: {xx[0]}')
    
    assert len(N80) == dx * dy, "N80 not the right size"
    N80_matrix = np.array(N80).reshape((dx, dy))
    
#Code lines to represent graphically N80 parameter in a colormap in the interface.   
  
    fig, axs =plt.subplots(1,1,dpi=100, figsize=(7,5), sharey=True)
    fig.suptitle('N80', size=20)
    
    # Define the colour scale and boundaries
    #cmap = 'viridis'
    #cmap=cm.get_cmap("Oranges")
    cmap=mcolors.LinearSegmentedColormap.from_list("magenta", ["white", "magenta"])
    vmin = min(N80)  #Minimum value of N80
    vmax = max(N80)  #Maximum value of N80
    
    x_ticks = np.linspace(-0.5, dy - 0.5, dy+1)
    y_ticks = np.linspace(-0.5, dx - 0.5, dx+1)
    
    x_labels = np.append(np.linspace(0, d_xaxis - (d_xaxis / dy), dy), d_xaxis)  # X-axis labels
    y_labels = np.append(np.linspace(0, d_yaxis - (d_yaxis / dx), dx), d_yaxis)  # Y-axis labels
    y_labels = y_labels[::-1]
    
    #Colour map created from N80_matrix
    
    plt.imshow(N80_matrix, cmap=cmap, aspect='auto', interpolation='bilinear', norm=Normalize(vmin=vmin, vmax=vmax))
    plt.xticks(ticks=x_ticks, labels=[f'{label:.1f} mm' for label in x_labels])
    plt.yticks(ticks=y_ticks, labels=[f'{label:.1f} mm' for label in y_labels])
    plt.colorbar(label='N80')
    
    
    canvas=FigureCanvasTkAgg(fig, master=frame2)
    canvas.draw()
    canvas.get_tk_widget().grid(column=4, row=1, rowspan=10)
    
    button10.config(bg="orange") #the colour of the button changes to indicate that this function has already been carried out.

"""
The test related to the button "SSpeed" is to obtain the t90 parameter for the polymer.
For this test, several potentials are applied to the polymer for a certain time. 
Therefore, first of all, a new window is generated so that the user can enter the time intervals related to the test. 
Subsequently, with this information, the contrasts of each time interval are calculated and finally, with these points, 
the function that is adapted to these points is generated. This function is used to obtain the parameter t90, 
which is finally plotted on a colour map. 
 
"""

def SSpeed():
    
    
    global s_pred #variable to store Transmittance spectra from the neural network. This variable is not neccesary to be global, but it can be useful to test in variable explorer
    global s_pred_reshaped # s_pred variable in other format to work later.This variable is not neccesary to be global, but it can be useful to test in variable explorer
    global data_for_nx #transmittance evolution of the images or frames at the wavelength given for the user
    global intervals #variable to store time intervals introduced for the user
    global time46 #the time to plot the transmittance evolution of the images or frames on the X-axis
    global intervals_contrast #variable to store contrast in each time interval
    global constants #variable to store constants of the generated function
    global x_values #Time intervals introduced for the user
    global array_intervals #Contrast values for each time interval
    global s #variable to know the method selected by the user. This variable is related to the "PDF" function
    global tc #t90 parameter obtained 
    global tc_matrix # t90 parameter store in a variable to graph in a colour map

    s=2 #It is associated with number 2 in "PDF" function
   
    #A new window is created for the user to enter the time intervals for the test.
    emergente1 = tk.Toplevel(frame2)
    emergente1.title("Switching Speed data")
    
    #Some variables are created to use in the next function
    intervalos = []
    interval_labels = []
    interval_entries = []
    
    #This function allows the generation of sufficient text boxes and labels for the number of intervals entered by the user. 
    def crear_intervalos():
        global num_intervalos
        
        try:
            num_intervalos = int(num_intervalos_var.get())
            if num_intervalos <= 0:
                raise ValueError("The number of intervals must be higher than 0.")
            
            start_row = len(interval_labels)
            for i in range(start_row, start_row + num_intervalos):  
                label = tk.Label(emergente1, text=f"Interval {i + 1}")
                label.grid(row=i + 2, column=0, padx=10, pady=5)
                entry = tk.Entry(emergente1)
                entry.grid(row=i + 2, column=1, padx=10, pady=5)
                label2 = tk.Label(emergente1, text="Number of repetitions")
                label2.grid(row=i + 2, column=2, padx=10, pady=5)
                entry1 = tk.Entry(emergente1)
                entry1.grid(row=i + 2, column=3, padx=10, pady=5)
                interval_labels.append(label)
                interval_entries.append(entry1)
                intervalos.append(entry)
            
            #This function is related to the "accept" button, so that all data entered by the user are saved in variables for later use.
            def cerrar_y_devolver():
                global valores_intervalos
                global valores_intervalos2
                try:
                    
                    valores_intervalos = [float(entry.get()) for entry in intervalos]
                    valores_intervalos2 = [float(entry1.get()) for entry1 in interval_entries]
                    print("Interval values:", valores_intervalos)
                    emergente1.destroy()
                except ValueError:
                    messagebox.showerror("Invalid entry", "Please, introduce valid numeric values")

            if len(interval_labels) == num_intervalos:
                boton_aceptar1 = tk.Button(emergente1, text="Accept", command=cerrar_y_devolver)
                boton_aceptar1.grid(row=start_row + num_intervalos + 2, columnspan=2, padx=10, pady=10)

        except ValueError as e:
            messagebox.showerror("Invalid entry", str(e))

    # architecture of the new window
    tk.Label(emergente1, text="Interval number for SS study:").grid(row=0, column=0, padx=10, pady=10)
    num_intervalos_var = tk.StringVar()
    tk.Entry(emergente1, textvariable=num_intervalos_var).grid(row=0, column=1, padx=10, pady=10)
    tk.Button(emergente1, text="Create intervals", command=crear_intervalos).grid(row=0, column=2, padx=10, pady=10)
    
    emergente1.grab_set()  # Blocks interaction with the main window, when the secondary window is closed then the main window will continue.
    emergente1.wait_window()
    
    #lamda3 store normalised wavelength introduced for the user
    s_pred=np.zeros((dt,n))
    
    if lamda==400:
        lamda3=0
    else:
        lamda3=(lamda-400)/300

    #loops to, firstly, isolate the Lab coordinates normalised to the wavelength entered by the user and 
    #,secondly, make the transmittance predictions from these Lab coordinates. 
    #In the variable data_for_nx is stored this evolution in the correct format.

    Labim_norm5=np.zeros((n,dt,4))

    for i in range(n):
        z=0
        for j in range(dt*nx):
            if Labim_norm4[i,j,3]==lamda3:
                Labim_norm5[i,z,0]=Labim_norm4[i,j,0]
                Labim_norm5[i,z,1]=Labim_norm4[i,j,1]
                Labim_norm5[i,z,2]=Labim_norm4[i,j,2]
                Labim_norm5[i,z,3]=Labim_norm4[i,j,3]
                z=z+1

    for i in range(n):
        
        prediccion = sol_n_n.predict(Labim_norm5[i,:,:])
        prediccion = prediccion.flatten()
        s_pred[:,i] = prediccion*100
        

    assert s_pred.shape == (dt, n), "s_pred does not have the expected dimensions"
    s_pred_reshaped = s_pred.T.reshape(n, dt)

    data_for_nx = s_pred_reshaped
    
    time46 = np.arange(1, n + 1) * time #variable to store time of the transmittance evolution
    
    
    intervals46=np.zeros(num_intervalos+1)
    contador=0
    Interval_values1=np.array(valores_intervalos) #Time interval values introduced for the user are store in this array
    Interval_values2=np.array(valores_intervalos2) #Number of repetitions for each interval values introduced for the user are store in this array
    intervals46[0]=0
    for i in range(num_intervalos):
        intervals46[i+1]=Interval_values1[i]*Interval_values2[i]*2+contador
        contador=intervals46[i+1]
        
    intervals=intervals46.tolist() #this variable store total time for each interval, taking into account each time interval and number of repetitions
    
    intervals_contrast = []
    
    for i in range(dt):
        #The transmittance evolution over time is plotted in the "plots" window.
        plt.figure(figsize=(10, 5))
        plt.plot(time46, data_for_nx[:, i])
        plt.title(f'Transmittance evolution of area {i+1}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Transmittance (%)')
        plt.legend()
        plt.grid(True)
        
        
        column_contrast = []
        
        #separation of each time interval and graphical representation of this separation
        
        for j in range(len(intervals) - 1):
            start_idx = np.searchsorted(time46, intervals[j])
            end_idx = np.searchsorted(time46, intervals[j + 1])
            
            interval_data = data_for_nx[start_idx:end_idx, i]
            
            if interval_data.size > 0:  # Verify that interval_data is not empty
                max_val = np.max(interval_data)
                min_val = np.min(interval_data)
                contrast46=max_val-min_val
            else:
                max_val = min_val = None
            
            column_contrast.append(contrast46)
            
            # Add a vertical line to display the intervals
            plt.axvline(x=intervals[j], color='r', linestyle='--')
        
        # Add the last vertical line
        plt.axvline(x=intervals[-1], color='r', linestyle='--')
        
        plt.show()
        
        intervals_contrast.append(column_contrast)
    
    array_intervals = np.array(intervals_contrast)
    
    x_values = np.array(valores_intervalos)
    
    #function that adapts to the development of contrast in different time intervals
    
    def curve_function(x, a, b):
        return a * (1 - np.exp(-b * x))
    
    constants = [] #variable to store constants of the resultant functions
    
    #loop to detect the function with its constants of each part of the area and plot this function in the plots window
    for i in range(dt):
        y_values = array_intervals[i, :]
        
        # Fitting the curve
        popt, pcov = curve_fit(curve_function, x_values, y_values, p0=[1, 1])
        a, b = popt
        constants.append((a, b))
        
        # Graph the points and the fitted curve
        plt.figure(figsize=(10, 5))
        plt.scatter(x_values, y_values, label='Data points')
        plt.plot(x_values, curve_function(x_values, a, b), color='r', label=f'Fitted curve: y = {a:.2f} * (1 - exp(-{b:.2f} * x))')
        plt.title(f'Curve function of area {i+1}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Contrast (%)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    array_constants=np.array(constants)
    tc=np.zeros((dt))
    #calculation of the t90 parameter
    for i in range(dt):
        
        tc[i]=1/array_constants[i,1]
        
    tc_matrix = np.array(tc).reshape((dx, dy)) #t90 parameter store in the correct form to graph in a colormap
    
    #Code lines to represent graphically t90 parameter in a colormap in the interface.  
    
    fig, axs =plt.subplots(1,1,dpi=100, figsize=(7,5), sharey=True)
    fig.suptitle('Time constant', size=20)
    
    # Define the colour scale and boundaries
    #cmap = 'viridis'
    #cmap=cm.get_cmap("Oranges")
    cmap=mcolors.LinearSegmentedColormap.from_list("black", ["white", "black"])
    vmin = min(tc)  # Minimum value of t90
    vmax = max(tc)  # Maximum value of t90
    
    x_ticks = np.linspace(-0.5, dy - 0.5, dy+1)
    y_ticks = np.linspace(-0.5, dx - 0.5, dx+1)
    
    x_labels = np.append(np.linspace(0, d_xaxis - (d_xaxis / dy), dy), d_xaxis)  
    y_labels = np.append(np.linspace(0, d_yaxis - (d_yaxis / dx), dx), d_yaxis)
    y_labels = y_labels[::-1]
    
    #Colour map of T90 parameter
    
    plt.imshow(tc_matrix, cmap=cmap, aspect='auto', interpolation='bilinear', norm=Normalize(vmin=vmin, vmax=vmax))
    plt.xticks(ticks=x_ticks, labels=[f'{label:.1f} mm' for label in x_labels])
    plt.yticks(ticks=y_ticks, labels=[f'{label:.1f} mm' for label in y_labels])
    plt.colorbar(label='Time constant')
    
    canvas=FigureCanvasTkAgg(fig, master=frame2)
    canvas.draw()
    canvas.get_tk_widget().grid(column=4, row=1, rowspan=10)
    
    button3.config(bg="orange") #the colour of the button changes to indicate that this function has already been carried out.
    
"""
# This button allows the user to export the results generated from the selected analysis process.
# When clicked, it opens a new window that displays a list of available outputs (plots and data tables),
# which vary depending on the process previously executed (Transmittance, Contrast, Switching Speed, or Stability).
# The user can choose which results to save as PDF (for figures) or CSV (for raw numerical data).

"""
#Function associated with button 13. Clicking this button, the graph shown in the interface and related data are saved in two different file formats (PDF, .csv) 
def PDF():
    
    global file #variable to store the name that the user has introduced in text box 8
    file= text_box8.get()
    
    selections={}
    
    if s==1:
        
        
        def save_selections():
            # Save the state of the checkboxes in the 'selections' variable
            selections["option1"] = var_option1.get()
            selections["option2"] = var_option2.get()
            selections["option3"] = var_option3.get()
            selections["option4"] = var_option4.get()
            # Close the popup window
            popup_window.destroy()
            
            
    
        # Create the popup window
        popup_window = Toplevel()
        popup_window.title("Options")
    
        # Variables for the checkboxes
        var_option1 = tk.IntVar()
        var_option2 = tk.IntVar()
        var_option3 = tk.IntVar()
        var_option4 = tk.IntVar()
    
        tk.Label(popup_window, text="Information to add in PDF").grid(row=0, column=1, padx=10, pady=10)
        tk.Label(popup_window, text="Information to add in csv").grid(row=0, column=2, padx=10, pady=10)
        tk.Label(popup_window, text="Transmittance Spectra").grid(row=1, column=0, padx=10, pady=10)
        tk.Label(popup_window, text="Colormap of contrast").grid(row=2, column=0, padx=10, pady=10)
        # Create checkboxes
        tk.Checkbutton(popup_window, text="", variable=var_option1).grid(row=1, column=1, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option2).grid(row=2, column=1, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option3).grid(row=1, column=2, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option4).grid(row=2, column=2, padx=10, pady=10)
    
        # Button to save selections and close
        tk.Button(popup_window, text="Save and Continue", command=save_selections).grid(row=3, column=0, padx=10, pady=10)
    
        popup_window.grab_set()  # Block interaction with the main window
        popup_window.wait_window()  # Pause execution until the popup window is closed
        
        if selections.get("option1",0):
            
            def generate_pdf_with_spectra(wlength, espectros, dt, output_file= str(file)+"_spectra"+".pdf"):
                """
                Generates graphs with two spectra per graph and saves them in a PDF with two graphs per page.
            
                Parameters:
                    wlength (array): Array for the x-axis, shape (61, 1).
                    espectros (array): Array for the y-axis, shape (2*dt, 61).
                    dt (int): Number of pairs of spectra (2 spectra per graph).
                    output_file (str): Name of the output PDF file.
                """
                num_graphs = dt  # Total number of graphs
                with PdfPages(output_file) as pdf:
                    for i in range(0, num_graphs, 2):
                        # Create a new page with up to two graphs
                        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Two graphs side by side
                        axs = axs.flatten()  # Flatten for easy indexing
                        
                        for j in range(2):
                            graph_index = i + j
                            if graph_index < num_graphs:
                                ax = axs[j]
                                
                                # Select the rows corresponding to this graph
                                y1 = espectros[2 * graph_index, :]   # First spectrum
                                y2 = espectros[2 * graph_index + 1, :]  # Second spectrum
                                
                                # Plot both spectra
                                ax.plot(wlength, y1*100, label=f"Spectrum {2 * graph_index + 1}")
                                ax.plot(wlength, y2*100, label=f"Spectrum {2 * graph_index + 2}")
                                ax.set_ylim(0, 100)
                                ax.set_title(f"Graph {graph_index + 1}")
                                ax.set_xlabel("Wavelength (nm)")
                                ax.set_ylabel("Transmittance (%)")
                                ax.legend()
                                ax.grid(True)
                            else:
                                axs[j].axis('off')  # Hide empty subplot if no more graphs
            
                        pdf.savefig(fig)  # Save the current page to the PDF
                        plt.close(fig)  # Free memory by closing the figure

                print(f"PDF generated: {output_file}")
            
            
            generate_pdf_with_spectra(wlength, espectros, dt)
        
        if selections.get("option2",0):
            
            def save_color_map_pdf(contrast46_matrix, contrast46, dx, dy, d_xaxis, d_yaxis, output_file=str(file)+"_contrast_colormap.pdf"):
                fig, axs = plt.subplots(1, 1, dpi=100, figsize=(7, 5), sharey=True)
                fig.suptitle('Contrast', size=20)
                
                x_ticks = np.linspace(-0.5, dy - 0.5, dy + 1)
                y_ticks = np.linspace(-0.5, dx - 0.5, dx + 1)
                x_labels = np.append(np.linspace(0, d_xaxis - (d_xaxis / dy), dy), d_xaxis)
                y_labels = np.append(np.linspace(0, d_yaxis - (d_yaxis / dx), dx), d_yaxis)
                y_labels = y_labels[::-1]
                
                # Define the color scale and limits
                
                cmap=mcolors.LinearSegmentedColormap.from_list("black", ["white", "black"])
                vmin = min(contrast46)  # Minimum value
                vmax = max(contrast46)  # Maximum value
                
                im = axs.imshow(
                    contrast46_matrix, cmap=cmap, aspect='auto', interpolation='bilinear',
                    norm=Normalize(vmin=vmin, vmax=vmax)
                )
                
                # Axis and label settings
                axs.set_xticks(x_ticks)
                axs.set_xticklabels([f'{label:.1f} mm' for label in x_labels], rotation=45)
                axs.set_yticks(y_ticks)
                axs.set_yticklabels([f'{label:.1f} mm' for label in y_labels])
                
                # Colour bar
                cbar = plt.colorbar(im, ax=axs, label='Contrast (%)')
                
                # Save the chart as a PDF
                with PdfPages(output_file) as pdf:
                    pdf.savefig(fig)
                    plt.close(fig)  # Close the figure to free up memory
                
                print(f"Colormap saved to: {output_file}")

            # Call to the function to save
            save_color_map_pdf(contrast46_matrix, contrast46, dx, dy, d_xaxis, d_yaxis)
        
        
        
        if selections.get("option3", 0):
            print("Spectra saved in csv")
            # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
            df1 = pd.DataFrame(wlength)
            df2 = pd.DataFrame(np.transpose(espectros))
    
            # Combines the DataFrames horizontally into a single DataFrame.
            df_combined = pd.concat([df1, df2], axis=1)
    
            # Specify the name of the CSV file that you want to create.
            file_name = str(file)+"_spectra"+".csv"
    
            # Save the combined DataFrame in a CSV file.
            df_combined.to_csv(file_name, index=False, header=False)
            
        if selections.get("option4", 0):
            print("Contrast colormap saved in csv")
            # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
            dt1=np.zeros(dt)
            for i in range(dt):
                dt1[i]=i+1
            df1 = pd.DataFrame(dt1)
            df2 = pd.DataFrame(contrast46)
    
            # Combines the DataFrames horizontally into a single DataFrame.
            df_combined = pd.concat([df1, df2], axis=1)
    
            # Specify the name of the CSV file that you want to create.
            file_name = str(file)+"_contrast_colormap"+".csv"
    
            # Save the combined DataFrame in a CSV file.
            df_combined.to_csv(file_name, index=False, header=False)
            
        saved.set("Saved")
        
        
    if s==2:
               
        def save_selections():
            # Save the state of the checkboxes in the 'selections' variable
            selections["option1"] = var_option1.get()
            selections["option2"] = var_option2.get()
            selections["option3"] = var_option3.get()
            selections["option4"] = var_option4.get()
            selections["option5"] = var_option5.get()
            selections["option6"] = var_option6.get()
            # Close the popup window
            popup_window.destroy()
    
        # Create the popup window
        popup_window = Toplevel()
        popup_window.title("Options")
    
        # Variables for the checkboxes
        var_option1 = tk.IntVar()
        var_option2 = tk.IntVar()
        var_option3 = tk.IntVar()
        var_option4 = tk.IntVar()
        var_option5 = tk.IntVar()
        var_option6 = tk.IntVar()
    
        tk.Label(popup_window, text="Information to add in PDF").grid(row=0, column=1, padx=10, pady=10)
        tk.Label(popup_window, text="Information to add in csv").grid(row=0, column=2, padx=10, pady=10)
        tk.Label(popup_window, text="Transmittance Evolution").grid(row=1, column=0, padx=10, pady=10)
        tk.Label(popup_window, text="Fitted function").grid(row=2, column=0, padx=10, pady=10)
        tk.Label(popup_window, text="Colormap of Switching speed parameter").grid(row=3, column=0, padx=10, pady=10)
        # Create checkboxes
        tk.Checkbutton(popup_window, text="", variable=var_option1).grid(row=1, column=1, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option2).grid(row=2, column=1, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option3).grid(row=3, column=1, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option4).grid(row=1, column=2, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option5).grid(row=2, column=2, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option6).grid(row=3, column=2, padx=10, pady=10)
    
        # Button to save selections and close
        tk.Button(popup_window, text="Save and Continue", command=save_selections).grid(row=4, column=0, padx=10, pady=10)
    
        popup_window.grab_set()  # Block interaction with the main window
        popup_window.wait_window()  # Pause execution until the popup window is closed
        
        if selections.get("option1",0):
            
            output_pdf=str(file)+"_transmittance_evolution"+".pdf"
            with PdfPages(output_pdf) as pdf:
                for i in range(dt):
                    # Create the figure
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(time46, data_for_nx[:, i])
                    ax.set_title(f'Transmittance evolution area {i+1}')
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('Transmittance (%)')
                    ax.set_ylim(0, 100)
                    ax.legend()
                    ax.grid(True)
                    
                    column_contrast = []
                    
                    for j in range(len(intervals) - 1):
                        start_idx = np.searchsorted(time46, intervals[j])
                        end_idx = np.searchsorted(time46, intervals[j + 1])
                        
                        interval_data = data_for_nx[start_idx:end_idx, i]
                        
                        if interval_data.size > 0:  # Verify that interval_data is not empty
                            max_val = np.max(interval_data)
                            min_val = np.min(interval_data)
                            contrast4 = max_val - min_val
                        else:
                            contrast4 = None
                        
                        column_contrast.append(contrast4)
                        
                        # Add a vertical line to display the intervals
                        ax.axvline(x=intervals[j], color='r', linestyle='--')
                    
                    # Add the last vertical line
                    ax.axvline(x=intervals[-1], color='r', linestyle='--')
                    
                    # Save the figure in the PDF
                    pdf.savefig(fig)
                    plt.close(fig)
            
                # Adjust the PDF pages to include 2 graphs per page
                pdf.infodict()['Title'] = 'Gráficas Generadas'
                pdf.infodict()['Author'] = 'Tu Nombre'

            print(f"PDF generated with success: {output_pdf}")
            
        if selections.get("option2",0):
            
            output_pdf=str(file)+"_curve_function"+".pdf"
            def curve_function(x, a, b):
                return a * (1 - np.exp(-b * x))
            
            with PdfPages(output_pdf) as pdf:
                constants = []  # List for storing adjustment constants
            
                for i in range(dt):
                    y_values = array_intervals[i, :]
            
                    # Fit the curve
                    popt, pcov = curve_fit(curve_function, x_values, y_values, p0=[1, 1])
                    a, b = popt
                    constants.append((a, b))
            
                    # Create the figure
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.scatter(x_values, y_values, label='Data points', color='blue')
                    ax.plot(x_values, curve_function(x_values, a, b), color='red', 
                            label=f'Curve fit: y = {a:.2f} * (1 - exp(-{b:.2f} * x))')
                    ax.set_title(f'Curve function of area {i+1}')
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('Contrast (%)')
                    ax.legend()
                    ax.grid(True)
            
                    # Save the figure in PDF
                    pdf.savefig(fig)
                    plt.close(fig)  # Close the figure to free up memory
            
                # Additional information in PDF
                pdf.infodict()['Title'] = 'Ajuste de Curvas'
                pdf.infodict()['Author'] = 'Tu Nombre'
            
            print(f"PDF generated with success: {output_pdf}")
        
        if selections.get("option3",0):
            
            def save_color_map_pdf(tc_matrix, tc, dx, dy, d_xaxis, d_yaxis, output_file=str(file)+"_speed_colormap.pdf"):
                fig, axs = plt.subplots(1, 1, dpi=100, figsize=(7, 5), sharey=True)
                fig.suptitle('Time constant', size=20)
                
                x_ticks = np.linspace(-0.5, dy - 0.5, dy + 1)
                y_ticks = np.linspace(-0.5, dx - 0.5, dx + 1)
                x_labels = np.append(np.linspace(0, d_xaxis - (d_xaxis / dy), dy), d_xaxis)
                y_labels = np.append(np.linspace(0, d_yaxis - (d_yaxis / dx), dx), d_yaxis)
                y_labels = y_labels[::-1]
                
                # Define the color scale and limits
                
                cmap=mcolors.LinearSegmentedColormap.from_list("black", ["white", "black"])
                vmin = min(tc)  # Minimum value
                vmax = max(tc)  # Maximum value
                
                im = axs.imshow(
                    tc_matrix, cmap=cmap, aspect='auto', interpolation='bilinear',
                    norm=Normalize(vmin=vmin, vmax=vmax)
                )
                
                # Axis and labels settings
                axs.set_xticks(x_ticks)
                axs.set_xticklabels([f'{label:.1f} mm' for label in x_labels], rotation=45)
                axs.set_yticks(y_ticks)
                axs.set_yticklabels([f'{label:.1f} mm' for label in y_labels])
                
                # Colour bar
                cbar = plt.colorbar(im, ax=axs, label='Time constant (s)')
                
                # Save the graph in PDF
                with PdfPages(output_file) as pdf:
                    pdf.savefig(fig)
                    plt.close(fig)  # Close the figure to free up memory
                
                print(f"Color map saved to: {output_file}")

            # Call to the function to save
            save_color_map_pdf(tc_matrix, tc, dx, dy, d_xaxis, d_yaxis)
        
        if selections.get("option4", 0):
            print("Transmittance evolution saved in csv")
            # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
            df1 = pd.DataFrame(time46)
            df2 = pd.DataFrame(data_for_nx)
    
            # Combines the DataFrames horizontally into a single DataFrame.
            df_combined = pd.concat([df1, df2], axis=1)
    
            # Specify the name of the CSV file that you want to create.
            file_name = str(file)+"_Transmittance_evolution_speed"+".csv"
    
            # Save the combined DataFrame in a CSV file.
            df_combined.to_csv(file_name, index=False, header=False) 

        if selections.get("option5", 0):
            print("Curve fitting points saved in csv")
            # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
            df1 = pd.DataFrame(x_values)
            df2 = pd.DataFrame(np.transpose(array_intervals))
    
            # Combines the DataFrames horizontally into a single DataFrame.
            df_combined = pd.concat([df1, df2], axis=1)
    
            # Specify the name of the CSV file that you want to create.
            file_name = str(file)+"_curve_fitting_points"+".csv"
    
            # Save the combined DataFrame in a CSV file.
            df_combined.to_csv(file_name, index=False, header=False)

        if selections.get("option6", 0):
            print("Colormap points saved in csv")
            # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
            dt1=np.zeros(dt)
            for i in range(dt):
                dt1[i]=i+1
            df1 = pd.DataFrame(dt1)
            df2 = pd.DataFrame(tc)
    
            # Combines the DataFrames horizontally into a single DataFrame.
            df_combined = pd.concat([df1, df2], axis=1)
    
            # Specify the name of the CSV file that you want to create.
            file_name = str(file)+"_speed_colormap"+".csv"
    
            # Save the combined DataFrame in a CSV file.
            df_combined.to_csv(file_name, index=False, header=False)                   
        
        
        saved.set("Saved")  
    
    if s==3:
        
        
        def save_selections():
            # Save the state of the checkboxes in the 'selections' variable
            selections["option1"] = var_option1.get()
            selections["option2"] = var_option2.get()
            selections["option3"] = var_option3.get()
            selections["option4"] = var_option4.get()
            selections["option5"] = var_option5.get()
            selections["option6"] = var_option6.get()
            # Close the popup window
            popup_window.destroy()
            
        # Create the popup window
        popup_window = Toplevel()
        popup_window.title("Options")
    
        # Variables for the checkboxes
        var_option1 = tk.IntVar()
        var_option2 = tk.IntVar()
        var_option3 = tk.IntVar()
        var_option4 = tk.IntVar()
        var_option5 = tk.IntVar()
        var_option6 = tk.IntVar()
    
        tk.Label(popup_window, text="Information to add in PDF").grid(row=0, column=1, padx=10, pady=10)
        tk.Label(popup_window, text="Information to add in csv").grid(row=0, column=2, padx=10, pady=10)
        tk.Label(popup_window, text="Transmittance Evolution").grid(row=1, column=0, padx=10, pady=10)
        tk.Label(popup_window, text="Fitted function").grid(row=2, column=0, padx=10, pady=10)
        tk.Label(popup_window, text="Colormap of stability parameter").grid(row=3, column=0, padx=10, pady=10)
        # Create checkboxes
        tk.Checkbutton(popup_window, text="", variable=var_option1).grid(row=1, column=1, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option2).grid(row=2, column=1, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option3).grid(row=3, column=1, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option4).grid(row=1, column=2, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option5).grid(row=2, column=2, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option6).grid(row=3, column=2, padx=10, pady=10)
    
        # Button to save selections and close
        tk.Button(popup_window, text="Save and Continue", command=save_selections).grid(row=4, column=0, padx=10, pady=10)
    
        popup_window.grab_set()  # Block interaction with the main window
        popup_window.wait_window()  # Pause execution until the popup window is closed
        
        if selections.get("option1", 0):
            
            output_pdf=str(file)+"_transmittance_evolution_stability"+".pdf"
            
            with PdfPages(output_pdf) as pdf:
                for i in range(dt):
                    # Create a new figure for each plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot the data and envelopes
                    ax.plot(np.arange(n) * 0.5, saved_data["y_values"][i], label="Transmittance evolution", color="blue")
                    ax.plot(np.arange(n) * 0.5, saved_data["envelope_upper"][i], label="Upper Envelope", color="green")
                    ax.plot(np.arange(n) * 0.5, saved_data["envelope_lower"][i], label="Lower Envelope", color="red")
                    #ax.fill_between(np.arange(n) * 0.5, saved_data["envelope_lower"][i], saved_data["envelope_upper"][i], color="yellow", alpha=0.3, label="Area Between Envelopes")
                    
                    # Set labels and title
                    ax.legend()
                    ax.set_title(f"Transmittance Evolution of area {i+1}")
                    ax.set_xlabel("Number of Cycles")
                    ax.set_ylabel("Transmittance (%)")
                    ax.set_ylim(0, 100)
                    
                    # Save the figure to the PDF
                    pdf.savefig(fig)
                    plt.close(fig)  # Close the figure to free memory
                    
        if selections.get("option2", 0):
            
            output_pdf=str(file)+"_fitting_curve_stability"+".pdf"
            
            def fitting_function(x, A1, t1, y0):
                return A1 * np.exp(-x / t1) + y0
            
            with PdfPages(output_pdf) as pdf:
                for i in range(dt):
                    # Create a new figure for each plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    diff_curve = saved_data["diff_curves"][i]
                    x_data = saved_data["x_data"][i]
                    fit_params = saved_data["fit_params"][i]
                    fitted_curve = fitting_function(x_data, *fit_params)
                    # Plot the data and envelopes
                    ax.plot(x_data, diff_curve, label="Differential curve", color="blue")
                    ax.plot(x_data, fitted_curve, label="Fitted curve", color="orange")
                    
                    
                    # Set labels and title
                    ax.legend()
                    ax.set_title(f"Fitted curve of area {i+1}")
                    ax.set_xlabel("Number of Cycles")
                    ax.set_ylabel("Contrast (%)")
                    ax.set_ylim(0, 100)
                    
                    # Save the figure to the PDF
                    pdf.savefig(fig)
                    plt.close(fig)  # Close the figure to free memory
          
        if selections.get("option3", 0):
            
            def save_color_map_pdf(N80_matrix, N80, dx, dy, d_xaxis, d_yaxis, output_file=str(file)+"_stability_colormap.pdf"):
                fig, axs = plt.subplots(1, 1, dpi=100, figsize=(7, 5), sharey=True)
                fig.suptitle('N80 parameter', size=20)
                
                x_ticks = np.linspace(-0.5, dy - 0.5, dy + 1)
                y_ticks = np.linspace(-0.5, dx - 0.5, dx + 1)
                x_labels = np.append(np.linspace(0, d_xaxis - (d_xaxis / dy), dy), d_xaxis)
                y_labels = np.append(np.linspace(0, d_yaxis - (d_yaxis / dx), dx), d_yaxis)
                y_labels = y_labels[::-1]
                
                # Define the color scale and limits
                
                cmap=mcolors.LinearSegmentedColormap.from_list("black", ["white", "black"])
                vmin = min(N80)  # Minimum value
                vmax = max(N80)  # Maximum value
                
                im = axs.imshow(
                    N80_matrix, cmap=cmap, aspect='auto', interpolation='bilinear',
                    norm=Normalize(vmin=vmin, vmax=vmax)
                )
                
                # Axis and labels settings
                axs.set_xticks(x_ticks)
                axs.set_xticklabels([f'{label:.1f} mm' for label in x_labels], rotation=45)
                axs.set_yticks(y_ticks)
                axs.set_yticklabels([f'{label:.1f} mm' for label in y_labels])
                
                # Colour bar
                cbar = plt.colorbar(im, ax=axs, label='N80 parameter')
                
                # Save the graph in PDF
                with PdfPages(output_file) as pdf:
                    pdf.savefig(fig)
                    plt.close(fig)  # Close the figure to free up memory
                
                print(f"Color map saved to: {output_file}")

            # Call to the function to save
            save_color_map_pdf(N80_matrix, N80, dx, dy, d_xaxis, d_yaxis)
        
        if selections.get("option4", 0):
            print("Transmittance evolution saved in csv")
            # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
            time47=np.zeros(n)
            for i in range(n):
                time47[i]=0.5*i+0.5
            df1 = pd.DataFrame(time47)
            df2 = pd.DataFrame(data_for_nx)
    
            # Combines the DataFrames horizontally into a single DataFrame.
            df_combined = pd.concat([df1, df2], axis=1)
    
            # Specify the name of the CSV file that you want to create.
            file_name = str(file)+"_Transmittance_evolution_stability"+".csv"
    
            # Save the combined DataFrame in a CSV file.
            df_combined.to_csv(file_name, index=False, header=False)
        
        if selections.get("option5", 0):
            print("Differential curve saved in csv")
            # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
            time47=np.zeros(n)
            for i in range(n):
                time47[i]=0.5*i+0.5
            
            df1 = pd.DataFrame(time47)
            df2 = pd.DataFrame(diff_curves_array)
    
            # Combines the DataFrames horizontally into a single DataFrame.
            df_combined = pd.concat([df1, df2], axis=1)
    
            # Specify the name of the CSV file that you want to create.
            file_name = str(file)+"_differential_curve_stability"+".csv"
    
            # Save the combined DataFrame in a CSV file.
            df_combined.to_csv(file_name, index=False, header=False)
            
        if selections.get("option6", 0):
            print("Colormap points saved in csv")
            # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
            dt1=np.zeros(dt)
            for i in range(dt):
                dt1[i]=i+1
            df1 = pd.DataFrame(dt1)
            df2 = pd.DataFrame(N80)
    
            # Combines the DataFrames horizontally into a single DataFrame.
            df_combined = pd.concat([df1, df2], axis=1)
    
            # Specify the name of the CSV file that you want to create.
            file_name = str(file)+"_stability_colormap"+".csv"
    
            # Save the combined DataFrame in a CSV file.
            df_combined.to_csv(file_name, index=False, header=False)
        
        saved.set("Saved")  
    
    if s==4:
        
        
        def save_selections():
            # Save the state of the checkboxes in the 'selections' variable
            selections["option1"] = var_option1.get()
            selections["option2"] = var_option2.get()
            selections["option3"] = var_option3.get()
            selections["option4"] = var_option4.get()
            selections["option5"] = var_option5.get()
            selections["option6"] = var_option6.get()
            # Close the popup window
            popup_window.destroy()
            
            
    
        # Create the popup window
        popup_window = Toplevel()
        popup_window.title("Options")
    
        # Variables for the checkboxes
        var_option1 = tk.IntVar()
        var_option2 = tk.IntVar()
        var_option3 = tk.IntVar()
        var_option4 = tk.IntVar()
        var_option5 = tk.IntVar()
        var_option6 = tk.IntVar()
    
        tk.Label(popup_window, text="Information to add in PDF").grid(row=0, column=1, padx=10, pady=10)
        tk.Label(popup_window, text="Information to add in csv").grid(row=0, column=2, padx=10, pady=10)
        tk.Label(popup_window, text="Transmittance Spectra").grid(row=1, column=0, padx=10, pady=10)
        tk.Label(popup_window, text="Colormap of Transmittance").grid(row=2, column=0, padx=10, pady=10)
        tk.Label(popup_window, text="Colormap of colorimetry").grid(row=3, column=0, padx=10, pady=10)
        # Create checkboxes
        tk.Checkbutton(popup_window, text="", variable=var_option1).grid(row=1, column=1, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option2).grid(row=2, column=1, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option3).grid(row=1, column=2, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option4).grid(row=2, column=2, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option5).grid(row=3, column=1, padx=10, pady=10)
        tk.Checkbutton(popup_window, text="", variable=var_option6).grid(row=3, column=2, padx=10, pady=10)
    
        # Button to save selections and close
        tk.Button(popup_window, text="Save and Continue", command=save_selections).grid(row=4, column=0, padx=10, pady=10)
    
        popup_window.grab_set()  # Block interaction with the main window
        popup_window.wait_window()  # Pause execution until the popup window is closed
        
        if selections.get("option1",0):
            
            def generate_pdf_with_spectra(wlength, espectros, dt, output_file= str(file)+"_spectra"+".pdf"):
                """
                Generates graphs with two spectra per graph and saves them in a PDF with two graphs per page.
            
                Parameters:
                    wlength (array): Array for the x-axis, shape (61, 1).
                    espectros (array): Array for the y-axis, shape (2*dt, 61).
                    dt (int): Number of pairs of spectra (2 spectra per graph).
                    output_file (str): Name of the output PDF file.
                """
                num_graphs = dt  # Total number of graphs
                with PdfPages(output_file) as pdf:
                    for i in range(num_graphs):
                        # Create a new page with up to two graphs
                        fig, ax = plt.subplots(figsize=(6, 6))  # Two graphs side by side
                        
                                
                                # Select the rows corresponding to this graph
                        y1 = espectros[i, :]   # First spectrum
                       
                        
                        # Plot both spectra
                        ax.plot(wlength, y1*100, label=f"Spectrum {i+ 1}")
                        
                        ax.set_ylim(0, 100)
                        ax.set_title(f"Graph {i + 1}")
                        ax.set_xlabel("Wavelength (nm)")
                        ax.set_ylabel("Transmittance (%)")
                        ax.legend()
                        ax.grid(True)
                    
    
                        pdf.savefig(fig)  # Save the current page to the PDF
                        plt.close(fig)  # Free memory by closing the figure

                print(f"PDF generated: {output_file}")
            
            
            generate_pdf_with_spectra(wlength, espectros, dt)
        
        if selections.get("option2",0):
            
            def save_color_map_pdf(transmittance46_matrix, transmittance46, dx, dy, d_xaxis, d_yaxis, output_file=str(file)+"_Transmittance_colormap.pdf"):
                fig, axs = plt.subplots(1, 1, dpi=100, figsize=(7, 5), sharey=True)
                fig.suptitle('Transmittance (%)', size=20)
                
                x_ticks = np.linspace(-0.5, dy - 0.5, dy + 1)
                y_ticks = np.linspace(-0.5, dx - 0.5, dx + 1)
                x_labels = np.append(np.linspace(0, d_xaxis - (d_xaxis / dy), dy), d_xaxis)
                y_labels = np.append(np.linspace(0, d_yaxis - (d_yaxis / dx), dx), d_yaxis)
                y_labels = y_labels[::-1]
                
                # Define the color scale and limits
                
                cmap=mcolors.LinearSegmentedColormap.from_list("black", ["magenta", "black"])
                vmin = min(transmittance46)  # Minimum value
                vmax = max(transmittance46)  # Maximum value
                
                im = axs.imshow(
                    transmittance46_matrix, cmap=cmap, aspect='auto', interpolation='bilinear',
                    norm=Normalize(vmin=vmin, vmax=vmax)
                )
                
                # Axis and labels settings
                axs.set_xticks(x_ticks)
                axs.set_xticklabels([f'{label:.1f} mm' for label in x_labels], rotation=45)
                axs.set_yticks(y_ticks)
                axs.set_yticklabels([f'{label:.1f} mm' for label in y_labels])
                
                # Colour bar
                cbar = plt.colorbar(im, ax=axs, label='Transmittance (%)')
                
                # Save the graph in PDF
                with PdfPages(output_file) as pdf:
                    pdf.savefig(fig)
                    plt.close(fig)  # Close the figure to free up memory
                
                print(f"Colormap saved to: {output_file}")

            # Call to the function to save
            save_color_map_pdf(transmittance46_matrix, transmittance46, dx, dy, d_xaxis, d_yaxis)
        
        
        
        if selections.get("option3", 0):
            print("Spectra saved in csv")
            # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
            df1 = pd.DataFrame(wlength)
            df2 = pd.DataFrame(np.transpose(espectros))
    
            # Combines the DataFrames horizontally into a single DataFrame.
            df_combined = pd.concat([df1, df2], axis=1)
    
            # Specify the name of the CSV file that you want to create.
            file_name = str(file)+"_spectra"+".csv"
    
            # Save the combined DataFrame in a CSV file.
            df_combined.to_csv(file_name, index=False, header=False)
            
        if selections.get("option4", 0):
            print("Transmittance colormap saved in csv")
            # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
            dt1=np.zeros(dt)
            for i in range(dt):
                dt1[i]=i+1
            df1 = pd.DataFrame(dt1)
            df2 = pd.DataFrame(transmittance46)
    
            # Combines the DataFrames horizontally into a single DataFrame.
            df_combined = pd.concat([df1, df2], axis=1)
    
            # Specify the name of the CSV file that you want to create.
            file_name = str(file)+"_transmittance_colormap"+".csv"
    
            # Save the combined DataFrame in a CSV file.
            df_combined.to_csv(file_name, index=False, header=False)
            
        

        if selections.get("option5",0):
            
            def save_color_map_pdf(ae_matrix, ae, dx, dy, d_xaxis, d_yaxis, output_file=str(file)+"_ae_colormap.pdf"):
                fig, axs = plt.subplots(1, 1, dpi=100, figsize=(7, 5), sharey=True)
                fig.suptitle('Colorimetry study', size=20)
                
                x_ticks = np.linspace(-0.5, dy - 0.5, dy + 1)
                y_ticks = np.linspace(-0.5, dx - 0.5, dx + 1)
                x_labels = np.append(np.linspace(0, d_xaxis - (d_xaxis / dy), dy), d_xaxis)
                y_labels = np.append(np.linspace(0, d_yaxis - (d_yaxis / dx), dx), d_yaxis)
                y_labels = y_labels[::-1]
                
                # Define the color scale and limits
                
                cmap=mcolors.LinearSegmentedColormap.from_list("black", ["black", "white"])
                vmin = min(ae)  # Minimum value
                vmax = max(ae)  # Maximum value
                
                im = axs.imshow(
                    ae_matrix, cmap=cmap, aspect='auto', interpolation='bilinear',
                    norm=Normalize(vmin=vmin, vmax=vmax)
                )
                
                # Axis and labels
                axs.set_xticks(x_ticks)
                axs.set_xticklabels([f'{label:.1f} mm' for label in x_labels], rotation=45)
                axs.set_yticks(y_ticks)
                axs.set_yticklabels([f'{label:.1f} mm' for label in y_labels])
                
                # Colour bar
                cbar = plt.colorbar(im, ax=axs, label='Euclidean error')
                
                # Save the graph in PDF
                with PdfPages(output_file) as pdf:
                    pdf.savefig(fig)
                    plt.close(fig)  # Close the figure to free up memory
                
                print(f"Colormap saved to: {output_file}")

            # Call to the function to save
            save_color_map_pdf(ae_matrix, ae, dx, dy, d_xaxis, d_yaxis)

        if selections.get("option6", 0):
            print("ae colormap saved in csv")
            # dataframes are created to introduces values of the X and Y axis of the resulting graph. These dataframes are neccesary to create the ".csv" file
            dt1=np.zeros(dt)
            for i in range(dt):
                dt1[i]=i+1
            df1 = pd.DataFrame(dt1)
            df2 = pd.DataFrame(ae)
    
            # Combines the DataFrames horizontally into a single DataFrame.
            df_combined = pd.concat([df1, df2], axis=1)
    
            # Specify the name of the CSV file that you want to create.
            file_name = str(file)+"_ae_colormap"+".csv"
    
            # Save the combined DataFrame in a CSV file.
            df_combined.to_csv(file_name, index=False, header=False)
        
        saved.set("Saved")
        
"""
clear function to reset all system variables and buttons. In addition to resetting the graph that appears on the interface
"""
#Function associated with button 14. This function reset all text boxes of the interface
def clear():
    
    global time
    global lamda
    global dx2
    global dy2
    global dz2
    #All variables are reseted to 0
    time=0
    lamda=0
    dx2=0
    dy2=0
    dz2=0
    
    #All text boxes of the interface are reseted
    wave.set("")
    wave1.set("")
    tframes.set("")
    tframes1.set("")
    process.set("")
    saved.set("")
    example1.set("")
    
    btn_abrir_emergente.config(bg="black")
    button4.config(bg="black")
    button3.config(bg="black")
    button9.config(bg="black")
    button10.config(bg="black")
    button18.config(bg="black")
    
    #Graph of the interface is reseted
    fig, axs =plt.subplots(1,1,dpi=100, figsize=(7,5), sharey=True)
    fig.suptitle('')
    
    canvas=FigureCanvasTkAgg(fig, master=frame2)
    canvas.draw()
    canvas.get_tk_widget().grid(column=4, row=1, rowspan=10)

"""
function that opens a pop-up window to add extra information relevant to the regions 
into which you want to divide the active area of the device to be measured. 
In addition to adding the target transmittance (%) value for the device in its colored state.
"""

def abrir_ventana_emergente():
    # Create the pop up window
    
    emergente = tk.Toplevel(frame2)
    emergente.title("Extra data")
    
    # Variables to store introduced data
    dx1 = tk.StringVar()
    dy1 = tk.StringVar()
    dz1 = tk.StringVar()
    
    # Function to close the pop-up window and return the data
    def cerrar_y_devolver():
        global dx2
        global dy2
        global dz2
        try:
            dx2 = int(dx1.get())
            dy2 = int(dy1.get())
            dz2 = float(dz1.get())
            datos = (dx1.get(), dy1.get())
            print("Entered data:", datos)
            emergente.destroy()
        except ValueError:
            messagebox.showerror("Invalid entry", "Please enter valid numeric values.")
    
    

    # Create widgets in the pop up window
    tk.Label(emergente, text="Partitions of the area to be selected").grid(row=0, column=0, padx=10, pady=10)
    tk.Label(emergente, text="X-axis divisions").grid(row=1, column=0, padx=10, pady=10)
    tk.Entry(emergente, textvariable=dy1).grid(row=1, column=1, padx=10, pady=10)
    
    tk.Label(emergente, text="Y-axis divisions").grid(row=2, column=0, padx=10, pady=10)
    tk.Entry(emergente, textvariable=dx1).grid(row=2, column=1, padx=10, pady=10)
    
    tk.Label(emergente, text="Maximum contrast colored transmittance").grid(row=3, column=0, padx=10, pady=10)
    tk.Entry(emergente, textvariable=dz1).grid(row=3, column=1, padx=10, pady=10)
    
    tk.Button(emergente, text="Accept", command=cerrar_y_devolver).grid(row=4, columnspan=2, padx=10, pady=10)
    
    emergente.grab_set()  # Blocks interaction with the main window
    emergente.wait_window()
    
    btn_abrir_emergente.config(bg="orange")
    
#All next functions are associated with info buttons 
def show_explanation():
    explanation = """
    Type of files that can be processed by this spectrophotometer are videos or a sequence of images. Please choose one of them.    
    """
    messagebox.showinfo("File Type", explanation)

def show_explanation1():
    explanation = """
Upload file:
    
Click this button to upload either a video or a sequence of images.

- Video: Select the file from any folder.

- Images:

1. Capture images at regular intervals (e.g., 1 s).

2. Save them in a folder located in the same directory as the script.

3. Select the first image; the system will process all subsequent files.

- Extra Data:
    
Define how many regions the selected area will be divided into (X and Y axes).
Also enter the “maximum contrast colored transmittance” value (used for ΔE map; required, even if unused).  
    """
    messagebox.showinfo("Step 1", explanation)

def show_explanation2():
    explanation = """
    If video option is selected, please indicate the desired interval (in seconds) for data processing.
    
    If Image sequence is selected, please indicate the interval (in seconds) with which the images were taken.
    
    In both cases, please indicate this interval in the first text box and then click in “Enter data” button. Automatically, this data will be saved in the system.
    """
    messagebox.showinfo("Step 2", explanation)

def show_explanation3():
    explanation = """
- To select the area of the images or video to study, please follow the next steps:
        
1.	First, press the button “Select area”.
2.	Please, wait until an image of the selected file appears in a pop-up window.
3.	Then, with the left mouse button, please select the upper left corner of the desired area of study. 
4.	With the right mouse button, please select the lower right corner, forming a rectangle corresponding to the area to be studied.
5.	Press "esc" key. 
6.	The process could take some time, mainly if the set of images is big or the video length is large. Please, wait until the process finishes. When the task is done, the next message will appear: “Process completed”

- To define a real scale (1 cm) for the maps, follow these steps:
        
1.  Press the button “Select 1 cm”.
2.  Wait until the image appears.
3.  With the left mouse button, select the starting point of a segment known to measure exactly 1 cm (horizontal).
4.  With the right mouse button, select the ending point of that 1 cm segment.
5.  The system will calculate the pixel distance and use it to annotate real dimensions in the final maps.
    """
    messagebox.showinfo("Step 3", explanation)

def show_explanation4():
    explanation = """
Please enter the wavelength (in nanometers) at which the analysis will be performed. Follow the next steps: 

- Type the value in the input box (e.g., 620)
- Press “Enter data”
- Once entered correctly, the value will appear next to the button.

This value is required for all processes (Transmittance, contrast, switching speed, and stability).  
    """
    messagebox.showinfo("Step 4", explanation)

def show_explanation5():
    explanation = """
- Transmittance study: Shows the transmittance distribution at λ_max across the selected area.
No additional input is required.
    
- Contrast study: Displays the contrast (ΔT at λ_max) for each region, calculated from the colored and bleached states.
Just click to run the process.
    
- SS study: Requires chronoamperometric data with potential steps of different durations:

    1. Enter the number of pulse intervals, then click “Create intervals”.

    2. For each interval, specify the number of repetitions.

    3. Click “Accept” to generate the t₉₀ map.
    
- Stability study: Estimates cycling durability using image data from a multi-cycle experiment.
Click to obtain the N₈₀ map showing the degradation behavior across the surface.

Once completed, each button turns orange to confirm successful processing.
    """
    messagebox.showinfo("Step 5", explanation)

def show_explanation6():
    explanation = """
To save data or plots, enter a filename in the “Save as” field and click “Save file”. A new window will appear where you can choose what to export.

- Right-click on each item to select it (✓ will appear).
- PDF: saves the figure
- CSV: saves raw data

Available options depend on the selected process:

1. Transmittance study:

- Transmittance Spectra (400–700 nm)

- Transmittance map at λ_max

- ΔE color difference map

2. Contrast study:
    
- Bleached and colored trasnmittance spectra

- Contrast map (ΔT at λ_max)

3. Switching Speed study (t₉₀):

- Transmittance over time

- Contrast vs. pulse duration + fit

- t₉₀ map

4. Stability study(N₈₀):

- Transmittance over cycles

- Contrast decay + exponential fit

- N₈₀ map

All selected files will be saved with the chosen name in your local directory (same to the script).

The clear button deletes all data added while using the interface so that you can start processing new data.

    """
    messagebox.showinfo("Step 6", explanation)

#function associated with button "drop_down_menu"

def select_option(option):#This function let the user select video or image sequence
    global option1# Variable to save the selected option by the user
    k.config(text="Selected option: " + option)
    if option=="Video":
        option1=1
    
    if option=="Image_sequence":
        option1=2

options = ["Video", "Image_sequence"]


#------------------------interface configuration in tkinter----------------------------------------

padx=5 #x-axis separation of buttons and textboxes (mm)
pady=5 #y-axis separation of buttons and textboxes (mm)
background="yellowgreen" # Colour background of the interface

#configuration of the interface
frame2=Frame()
frame2.pack(fill="both",expand="True")
frame2.config(bg=background)
frame2.config(bd=35) #frame border width
frame2.config(relief="flat")  #for border, border type

#Configuration of the text that appears in the interface
Label(frame2, text="File Type", fg="black",bg=background,font=("arial",14)).grid(row=0, column=0, sticky="w", padx=10, pady=10)
Label(frame2, text="1. Browse: upload video or image sequence.", fg="black",bg=background,font=("arial",14)).grid(row=1, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="2. Select data acquisition interval.", fg="black",bg=background,font=("arial",14)).grid(row=3, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="3. Select area to be processed.", fg="black",bg=background,font=("arial",14)).grid(row=5, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="4. Select wavelength to study.", fg="black",bg=background,font=("arial",14)).grid(row=7, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="5. Select a process.", fg="black",bg=background,font=("arial",14)).grid(row=9, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="6. Save file as.", fg="black",bg=background,font=("arial",14)).grid(row=11, column=0,sticky="w",padx=10, pady=10,columnspan=3)
Label(frame2, text="Time (seconds)", fg="black",bg=background,font=("arial",14)).grid(row=4, column=0,padx=padx, pady=pady)
Label(frame2, text="Wavelength (nm)", fg="black",bg=background,font=("arial",14)).grid(row=8, column=0,padx=padx, pady=pady)

#Configuration of the arrow used in the interface in several buttons
arrow = "\u2192"

text_with_arrow = f"Enter data {arrow}"
#Configuration of all buttons that appear in the interface
button18=Button(frame2, text="Transmittance study", fg="white", bg="black",font=("arial",14), command=T_study)
button18.grid(row=10,column=0,padx=padx, pady=pady)

button3=Button(frame2, text="SS study", fg="white", bg="black",font=("arial",14), command=SSpeed)
button3.grid(row=10,column=2,padx=padx, pady=pady)

button4=Button(frame2, text="Select 1 cm", fg="white", bg="black",font=("arial",14), command=measure)
button4.grid(row=6,column=2,padx=padx, pady=pady)

button5=Button(frame2, text="Upload file", fg="white", bg="black",font=("arial",14), command=examine)
button5.grid(row=2,column=0,padx=padx, pady=pady)

button8=Button(frame2, text="Select Area", fg="white", bg="black",font=("arial",14), command=calculate)
button8.grid(row=6,column=0,padx=padx, pady=pady)  

button9=Button(frame2, text="Contrast study", fg="white", bg="black",font=("arial",14), command=CP)
button9.grid(row=10,column=1,padx=padx, pady=pady)

button10=Button(frame2, text="Stability study", fg="white", bg="black",font=("arial",14), command=SP)
button10.grid(row=10,column=3,padx=padx, pady=pady)

button13=Button(frame2, text="Save file", fg="white", bg="black",font=("arial",14), command=PDF)
button13.grid(row=12,column=1,padx=padx, pady=pady)

button14=Button(frame2, text="Clear", fg="white", bg="black",font=("arial",14), command=clear)
button14.grid(row=12,column=3,padx=padx, pady=pady)

button16=Button(frame2, text=text_with_arrow, fg="white", bg="black",font=("arial",14),command=timef)
button16.grid(row=4,column=1,sticky="e",padx=padx, pady=pady) 

button17=Button(frame2, text=text_with_arrow, fg="white", bg="black",font=("arial",14),command=landa)
button17.grid(row=8,column=1,sticky="e",padx=padx, pady=pady) 

btn_abrir_emergente =Button(frame2, text="Extra data",fg="white", bg="black",font=("arial",14), command=abrir_ventana_emergente)
btn_abrir_emergente.grid(row=2,column=2,padx=padx, pady=pady)


#Configuration of all information buttons

button_info = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info.grid(row=0,column=1,sticky="e",padx=0, pady=0)
button_info.bind("<Button-1>", lambda event: show_explanation())

button_info1 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info1.grid(row=1,column=1,sticky="e",padx=0, pady=0)
button_info1.bind("<Button-1>", lambda event: show_explanation1())

button_info2 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info2.grid(row=3,column=1,sticky="e",padx=0, pady=0)
button_info2.bind("<Button-1>", lambda event: show_explanation2())

button_info3 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info3.grid(row=5,column=1,sticky="e",padx=0, pady=0)
button_info3.bind("<Button-1>", lambda event: show_explanation3())

button_info4 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info4.grid(row=7,column=1,sticky="e",padx=0, pady=0)
button_info4.bind("<Button-1>", lambda event: show_explanation4())

button_info5 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info5.grid(row=9,column=1,sticky="e",padx=0, pady=0)
button_info5.bind("<Button-1>", lambda event: show_explanation5())

button_info6 = tk.Label(frame2, text="i", fg="black", cursor="hand2")
button_info6.grid(row=11,column=1,sticky="e",padx=0, pady=0)
button_info6.bind("<Button-1>", lambda event: show_explanation6())

#Configuration of the drop down menu in the interface
selected_option = tk.StringVar(frame2)
selected_option.set("")

drop_down_menu = tk.OptionMenu(frame2, selected_option, *options, command=select_option)
drop_down_menu.grid(row=0,column=1,sticky="w",padx=padx, pady=pady)

k = tk.Label(frame2)

#All variables, related with the text box of the interface, are named through StringVar variables

example1=StringVar()
process=StringVar()
saved=StringVar()
wave=StringVar()
tframes=StringVar()
wave1=StringVar()
tframes1=StringVar()

# Configuration of all the text boxes that appear in the interface
text_box3=Entry(frame2, textvariable=example1,font=("arial",14)) 
text_box3.grid(row=2, column=1,padx=padx, pady=pady)
text_box4=Entry(frame2, width=6,textvariable=tframes1,font=("arial",14))
text_box4.grid(row=4, column=1,sticky="w",padx=padx, pady=pady) 
text_box5=Entry(frame2,width=6,textvariable=wave1,font=("arial",14))
text_box5.grid(row=8, column=1,sticky="w",padx=padx, pady=pady)
text_box6=Entry(frame2, textvariable=process,font=("arial",14))
text_box6.grid(row=6, column=1,padx=padx, pady=pady) 
text_box8=Entry(frame2, textvariable=saved,font=("arial",14))
text_box8.grid(row=12, column=0,padx=padx, pady=pady)
text_box9=Entry(frame2, width=6,textvariable=tframes,font=("arial",14))
text_box9.grid(row=4, column=2,sticky="w",padx=50, pady=pady)
text_box10=Entry(frame2,width=6,textvariable=wave,font=("arial",14))
text_box10.grid(row=8, column=2,sticky="w",padx=50, pady=pady)

raiz.mainloop()

