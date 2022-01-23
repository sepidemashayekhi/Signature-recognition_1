import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def  extract_feture(directoty,count):
    shift = cv2.SIFT_create()
    feature=[]
    label=[]
    imagegen = ImageDataGenerator()
    imageGenerator=imagegen.flow_from_directory(directoty,target_size=(200,200),batch_size=1,class_mode='binary')

    i=0
    for image, label in imageGenerator:
        image=np.reshape(image,newshape=(200,200,3))
        image=np.asarray(image)
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGRA)
        image=image.astype('uint8')
        key,dst=shift.detectAndCompute(image,None)
        dst=np.reshape(dst,newshape=(-1,dst.shape[0]*dst.shape[1]))
        feature.append(dst)

        i+=1
        if i>=count:
            break
    return  feature



def thresh(image):
    image=np.reshape(image,newshape=(200,200,3))
    image=np.asarray(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(image,35,255,cv2.THRESH_BINARY)
    return thresh




