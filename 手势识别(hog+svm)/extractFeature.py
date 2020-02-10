# -*- coding:utf-8 -*-
# 提取图像的HOG特征

import cv2

train_path   = "./images/"
feature_path      = "./feature/"
test_path    = "./test_images/"
test_feature_path = "./test_feature/"
winSize = (256,256)
blockSize = (128,128)
blockStride = (32,32)
cellSize = (64,64)
nbins = 1
winStride = (128,128)
padding = (0,0)


def hog_features(res):
	hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins) # 设置HOG参数
	descriptors = hog.compute(res, winStride, padding).reshape((-1,)) # 计算HOG特征
	return descriptors


if __name__=="__main__":
	for i in range(1,5):
		for j in range(1, 355):
			#### 对训练图片
			train_res = cv2.imread(train_path+str(i)+"_"+str(j)+".png")
			train_dp = hog_features(train_res)
			train_dp_name = feature_path+str(i)+"_"+str(j)+".txt"
			with open(train_dp_name, "w", encoding="utf-8") as f:
				temp = train_dp[1]
				for k in range(1, len(train_dp)):
					d_record = int(100 * train_dp[k])
					f.write(str(d_record))
					f.write(" ")
				f.write("\n")
			print(train_dp_name, "完成")


			# #### 对测试图片
			# test_res = cv2.imread(test_path+str(i)+"_"+str(j)+".png")
			# test_dp = hog_features(test_res)
			# test_dp_name = test_feature_path+str(i)+"_"+str(j)+".txt"
			# with open(test_dp_name, "w", encoding="utf-8") as f:
			# 	temp = test_dp[1]
			# 	for k in range(1, len(test_dp)):
			# 		d_record = int(100 * test_dp[k])
			# 		f.write(str(d_record))
			# 		f.write(" ")
			# 	f.write("\n")
			# print(test_dp_name, "完成")