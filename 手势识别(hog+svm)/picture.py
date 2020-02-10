# -*- coding:utf-8 -*-
#对图像进行二值化和肤色检测

import cv2
import numpy as np 
import math
from extractFeature import hog_features
from scipy.ndimage.interpolation import geometric_transform


# 二值化处理
def binaryMask(frame, x0, y0, width, height):
	cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0,255,0)) # 画出录像时截图区域
	roi = frame[y0:y0+height, x0:x0+width] # 获取手势框图
	res = skinMask(roi) # 进行肤色检测
	hog_result = hog_features(res)
	return roi, res, hog_result


# 肤色检测
def skinMask(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) # 将RGB转至YCrCb
	(y, cr, cb) = cv2.split(YCrCb) # 拆分YCrCb各个颜色通道
	cr1 = cv2.GaussianBlur(cr, (5,5), 0)
	_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Ostu处理
	res = cv2.bitwise_and(roi, roi, mask=skin)
	return res