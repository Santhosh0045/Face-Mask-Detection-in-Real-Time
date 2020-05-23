import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
from PIL import Image
import os
from torchvision import models,transforms
from torch import optim
import torch.nn as nn

def create_model(n_classes):		
	model = models.resnet18(pretrained=True)

	n_features = model.fc.in_features

	model.fc = nn.Sequential(
	 	nn.Linear(n_features, n_classes),
	    nn.Softmax(dim=1)
	  )

	return model


print('Instantiating....')
mymodel = create_model(2)
print('Done!!')

print('Model Loading....')
mymodel.load_state_dict(torch.load('/home/santhosh/Desktop/Mask_Detection/model.pth'))
mymodel.eval()
print('Done!!')

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = transforms.Compose([
                  transforms.RandomResizedCrop(size=256),
                  transforms.RandomRotation(degrees=15),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize(mean, std)
])

cam = cv2.VideoCapture(0)


while(cam.isOpened()):
  
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5) 
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 

    for (x,y,w,h) in faces:
    
        face_img=image[y:y+h,x:x+h]
        pil_face = Image.fromarray(face_img)
        face_trans = transform(pil_face)
        face = face_trans.unsqueeze(0)
        result = mymodel(face)
        

        y_=torch.argmax(result,axis=1)
        label = y_.item()
        
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv2.LINE_AA)
        
        
  
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    if ret==False:
    	break
        
cv2.destroyAllWindows()
cam.release()
