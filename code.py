import numpy as np 
import pandas as pd 
import splitfolders
import cv2 as cv 
import argparse
from __future__ import print_function
import torch 
from torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

#splitting of the dataset(uncomment only once to split the folders )
drowsy_img = '/Users/birba/OneDrive/Documents/projects_github/drowsiness/Drowsiness_detction/dataset'
output_drow = '/Users/birba/OneDrive/Documents/projects_github/drowsiness/Drowsiness_detction/splitted_dataset'
#splitfolders.ratio(drowsy_img, output=output_drow, seed=1337, ratio=(0.8, 0.1, 0.1), group_prefix=None, move=False)

#splitfolders.ratio(nondrowsy_img, output=output_drow, seed=1337, ratio=(0.8, 0.1, 0.1), group_prefix=None, move=False)

#face detection in the image 
def detectdisplay (frame):
    frame_gray= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    frame_gray= cv.equalizeHist(frame_gray)
    
    faces = face_cascade.detectMultiscale(frame_gray)
    for(x,y,w,h) in faces:
        centre = (x+w//2,y+h//2)
        frame = cv.ellipse(frame,centre,(w//2,h//2),0,0,360,(255,0,255),4)
        
        frameROI= frame_gray[y:y+h,x:x+w]
        #eye detection
        eyes = eyes_cascade.detectMultiscale(frameROI)
        for (x2,y2,w2,h2) in eyes:
            centre_eye = (x+x2+w2//2,y+y2+h2//2)
            radius = int(round(w2+h2)*0.25)
            eye_frame = cv.circle(eye_frame,centre_eye,(w2//2,h2//2),radius,(255,0,0),4)
            
    cv.imshow('capture-face',frame)
    

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

#loading the cascades
if not face_cascade.load(cv.samples.Findfile(face_cascade_name)):
    print("error in loading faces")
    exit(0)
    
if not eyes_cascade.load(cv.samples.Findfile(eyes_cascade_name)):
    print("error in locating eyes")
    exit(0)
    
#loading images
test_split='/Users/birba/OneDrive/Documents/projects_github/drowsiness/Drowsiness_detction/splitted_dataset/test'
train_split='/Users/birba/OneDrive/Documents/projects_github/drowsiness/Drowsiness_detction/splitted_dataset/train'
validation_split='/Users/birba/OneDrive/Documents/projects_github/drowsiness/Drowsiness_detction/splitted_dataset/val'
# out_res='/Users/birba/OneDrive/Documents/projects_github/Drowsiness_detection-1/resized_image'
frame = cv.imread(train_split)
if frame is None:
    print("Loading error")
    exit(0)
    
#detection
detectdisplay(frame)

#image pre-processing

transform = transforms.Compose[(
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalise(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
)]

image = transform(frame)
 
patch_size = 16 
patches = image.unfold(1,patch_size,patch_size).unfold(2,patch_size,patch_size)
patches = patches.contiguous().view(3,-1,patch_size,patch_size)

#patch embeddings

def patchembeddings(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_channels=3,embedd_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection nn.Conv2d(in_channels,embedd_dim,kernel_size=patch_size,stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1,self.num_patches,embedd_dim))

    def forward(Self,x):
        x = self.projection(x)  # (B,embedd_dim,H/patch_size,W/patch_size)
        x = x.flatten(2)  # (B,embedd_dim,N)
        x = x.transpose(1,2)  # (B,N,embedd_dim)
        x = x + self.position_embeddings  # (B,N,embedd_dim)
        return x









