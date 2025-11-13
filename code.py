import numpy as np 
import pandas as pd 
import splitfolders
import cv2 as cv 
import argparse
from __future__ import print_function
import torch 
from torch import nn,optim
from torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets,models,transforms
from torchvision.models import vit_b_16


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

def patchembeddings (nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_channels=3,embedd_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels,embedd_dim,kernel_size=patch_size,stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1,self.num_patches,embedd_dim))

    def forward(Self,x):
        x = self.projection(x)  
        x = x.flatten(2)  
        x = x.transpose(1,2) 
        x = x + self.position_embeddings  
        return x

#trasnformer encodding 

def visionTransformer(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_channels=3,embedd_dim=768,num_heads=12,num_layers=12,mlp_dim=3072,num_classes=2,dropout_rate=0.1):
        super().__init__()
        self.patch_embeddings = patchembeddings(img_size,patch_size,in_channels,embedd_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedd_dim,nhead=num_heads,dim_feedforward=mlp_dim,dropout=dropout_rate,activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.classifier = nn.Linear(embedd_dim,num_classes)

    def forward(self,x):
        x = self.patch_embeddings(x)  
        x = x.transpose(0,1)  
        x = self.transformer_encoder(x)  
        x = x.mean(dim=0)  
        x = self.classifier(x)
        return x

#face feature extraction//classification head

def ClassificationHead(nn.module):
    def __init__(self,input_dim,num_classes,dropout_prob=0.1):
        super(ClassificationHead.self).__init__()
        self.dropout=nn.Dropout(dropout_prob)
        self.fc=nn.Linear(input_dim,num_classes)
    
    def forward(self,cls_token_output):
        x=self.dropout(cls_token_output)    
        logits=self.fc(x)
        return logits
    

def x``
