


from facenet_pytorch import MTCNN

from PIL import Image
import torch
import os
from tqdm import tqdm
import math
import cv2
import numpy as np
device=torch.device('cuda:0')

r_img=Image.open('test.jpg')


mtcnn = MTCNN(image_size=120)


boxes, probs, landmarks = mtcnn.detect(r_img, landmarks=True)

face_tensor, prob = mtcnn(r_img, save_path='face.png', return_prob=True)

img=np.array(r_img)

    
x1,y1,x2,y2=landmarks[0][3][0],landmarks[0][3][1],landmarks[0][4][0],landmarks[0][4][1]
    
    
eye_center = (x1+x2)/2,(y1+y2)/2
    
dx = abs(x2-x1)
dy = abs(y2-y1)

angle = -math.atan2(dy, dx) * 180. / math.pi  # 计算角度
RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵

align_face = cv2.warpAffine(img, RotateMatrix, (img.shape[0], img.shape[1]))  # 进行放射变换，即旋转

n_img=Image.fromarray(align_face)

face_tensor, prob = mtcnn(n_img, save_path='face2.png', return_prob=True)


s_n=Image.open('face.png')
s_n=np.array(s_n)

s_face = cv2.warpAffine(s_n,RotateMatrix,  (s_n.shape[0], s_n.shape[1]))  # 进行放射变换，即旋转


s_img=Image.fromarray(s_face.astype(np.uint8))
