# -*- coding:utf-8 -*-
# 拓展数据集

import random
import cv2
path = './' + 'up' + '/'
outpath = './'+'images'+'/'

#旋转
def rotate(image, scale=0.9):
    angle = random.randrange(-13, 13) # 随机角度
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image

if __name__ == "__main__":
	
