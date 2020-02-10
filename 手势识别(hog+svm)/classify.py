# -*- coding: utf-8 -*-
# 使用SVM对手势方向进行训练, 分类

import cv2
import numpy as np 
from os import listdir
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

train_path = "./feature/"
test_path  = "./test_feature/"
model_path = "./model/"

def txtToVector(filename, N):
	returnVec = np.zeros((1,N))
	fr = open(filename)
	lineStr = fr.readline()
	lineStr = lineStr.split(" ")
	for i in range(N):
		returnVec[0, i] = int(lineStr[i])
	return returnVec


# 训练SVM
def train_SVM(N):
	svc = SVC()
	parameters = {'kernel':('linear', 'rbf'),
	              'C':[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
	              'gamma':[0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}#预设置一些参数值
	hwLabels = []#存放类别标签
	trainingFileList = listdir(train_path)
	m = len(trainingFileList)
	trainingMat = np.zeros((m,N))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		hwLabels.append(classNumber)
		trainingMat[i,:] = txtToVector(train_path+fileNameStr, N)#将训练集改为矩阵格式
	clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)#网格搜索法，设置5-折交叉验证
	clf.fit(trainingMat, hwLabels)
	print(clf.return_train_score)
	print(clf.best_params_) #打印出最好的结果
	best_model = clf.best_estimator_
	print("SVM Model saved")
	save_path = model_path + "svm_train_model.m"
	joblib.dump(best_model,save_path)#保存最好的模型


# 测试SVM
def test_SVM(clf,N):
	testFileList = listdir(train_path)
	errorCount = 0#记录错误个数
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		classNum = int(fileNameStr.split('_')[0])
		vectorTest = txtToVector(train_path+fileNameStr,N)
		valTest = clf.predict(vectorTest)
		#print("分类返回结果为%d\t真实结果为%d" % (valTest, classNum))
		print("file:", fileNameStr,"classNum:", classNum, "Test:", clf.predict(vectorTest))
		if valTest != classNum:
			errorCount += 1
	print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))



def test_fd(fd_test):
	clf = joblib.load(model_path + "svm_train_model.m")
	test_svm = clf.predict(fd_test)
	return test_svm


if __name__=="__main__":
	train_SVM(70)
	clf = joblib.load(model_path + "svm_train_model.m")
	test_SVM(clf,70)



