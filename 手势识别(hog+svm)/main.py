 # -*- coding:utf-8 -*-
# svm + hog特征实现上下左手手势识别
# 返回手势数字：
# 1 - 右
# 2 - 左
# 3 - 上
# 4 - 下
# 0 - 无手势
# 环境版本：
# openCV => 3.4.5
# scikit-learn => 0.20.2
# numpy => 1.17.4
import cv2
import numpy as np 
from sklearn.externals import joblib

#调用此函数的到手势识别的数字,输入数据为 300 X 300的图像
def hanqingcode(img):
	count = 0
	res, hog_result = binaryMask(img)
	fd_test = np.zeros((1, 70))
	temp = hog_result[1]
	# print(hog_result)
	for k in range(1, 70):
		fd_test[0, k - 1] = int(100 * hog_result[k])
	for i in fd_test[0]:
		if i == 49:
			count += 1
	test_svm = test_fd(fd_test)
	if count>=60:
		return 0
	return test_svm[0]

# 肤色检测
def skinMask(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) # 将RGB转至YCrCb
	(y, cr, cb) = cv2.split(YCrCb) # 拆分YCrCb各个颜色通道
	cr1 = cv2.GaussianBlur(cr, (5,5), 0)
	_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Ostu处理
	res = cv2.bitwise_and(roi, roi, mask=skin)
	return res

# 二值化处理
def binaryMask(frame):
	# cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0,255,0)) # 画出录像时截图区域
	# roi = frame[y0:y0+height, x0:x0+width] # 获取手势框图
	res = skinMask(frame) # 进行肤色检测
	hog_result = hog_features(res)
	return res, hog_result

# 提取HOG特征
def hog_features(res):
	winSize = (256,256)
	blockSize = (128,128)
	blockStride = (32,32)
	cellSize = (64,64)
	nbins = 1
	winStride = (128,128)
	padding = (0,0)
	hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins) # 设置HOG参数
	descriptors = hog.compute(res, winStride, padding).reshape((-1,)) # 计算HOG特征
	return descriptors

# 加载模型并分类
def test_fd(fd_test):
	clf = joblib.load("./model/svm_train_model.m")
	test_svm = clf.predict(fd_test)
	return test_svm


###调用示例
if __name__=="__main__":
	cap = cv2.VideoCapture(0)
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret==True:
			frame = cv2.flip(frame, 1)
			h, w, c = frame.shape
			#裁取中心图像(300x300)
			frame = frame[h//2 - 150:h//2 + 150, w//2 - 150:w//2 + 150]
			############
			##传入图像调用
			d = hanqingcode(frame)
			##########
			if d == 1:
				print("右")
			elif d == 2:
				print("左")
			elif d == 3:
				print("上")
			elif d == 4:
				print("下")
			else:
				print("无手势")

			cv2.imshow("frame", frame)
			if cv2.waitKey(25) == ord('q'):
				cv2.destroyAllWindows()
				cap.release()
				break






