# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:35:24 2018


@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python. 
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper, 
   please contact the authors of related paper.
"""

'''

Installation has been tested with Python 3.5.
Since the package is written in python 3.5, 
python 3.5 with the pip tool must be installed first. 
It uses the following dependencies: numpy(1.16.3), scipy(1.2.1), keras(2.2.0), sklearn(0.20.3)  
You can install these packages first, by the following commands:

pip install numpy
pip install scipy
pip install keras (if use keras data_load())
pip install scikit-learn
'''
import numpy as np
import pandas as pd
import scipy.io as scio
from BroadLearningSystem import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes
from sklearn.model_selection import train_test_split
''' For Keras dataset_load()'''
# import keras
import os
from PIL import Image
import numpy as np
# (traindata, trainlabel), (testdata, testlabel) = keras.datasets.mnist.load_data()
# traindata = traindata.reshape(traindata.shape[0], 28*28).astype('float64')/255
# trainlabel = keras.utils.to_categorical(trainlabel, 10)
# testdata = testdata.reshape(testdata.shape[0], 28*28).astype('float64')/255
# testlabel = keras.utils.to_categorical(testlabel, 10)
# dataFile = './matlab1.mat'
# data = scio.loadmat(dataFile)
# traindata = np.double(data['features'])
# trainlabel = np.double(data['classes'])
# train_X,test_X,train_y,test_y = train_test_split(traindata12,trainlable12,test_size=0.1)
# dataFile = './matlab1.mat'
# data = scio.loadmat(dataFile)
# testdata = np.double(data['features'])
# testlabel = np.double(data['classes'])
# dataFile = './mnist.mat'
# data = scio.loadmat(dataFile)
# traindata = np.double(data['train_x']/255)
# trainlabel = np.double(data['train_y'])
# testdata = np.double(data['test_x']/255)
# testlabel = np.double(data['test_y'])
# dataFile = 'D:\\1xuexi\\daima\\naojijiekou\\test\\BLS_CODE_PYTHON1\\mnist.mat'
# data = scio.loadmat(dataFile)
# traindata = np.double(data['train_x']/255)
# trainlabel = np.double(data['train_y'])
# testdata = np.double(data['test_x']/255)
# testlabel = np.double(data['test_y'])
# train_X1,train_X2,train_y1,train_y2 = train_test_split(traindata,trainlabel,test_size=0.5,)
# from keras.utils import to_categorical
# 导入数据
#MNIST
# dataFile = './mnist.mat'
# data = scio.loadmat(dataFile)
# traindata = np.double(data['train_x']/255)
# trainlabel = np.double(data['train_y'])
# testdata = np.double(data['test_x']/255)
# testlabel = np.double(data['test_y'])
# train_X1,train_X2,train_y1,train_y2 = train_test_split(traindata,trainlabel,test_size=0.5,)
#意识障碍
# df = pd.read_csv('pdoc.csv')
# traindata1 = df.iloc[1:, :-10].to_numpy()
# trainlabel1_series = df.iloc[1:, -1]
# trainlabel1 = pd.get_dummies(trainlabel1_series).to_numpy()
# traindata,testdata ,trainlabel, testlabel=train_test_split(traindata1, trainlabel1, train_size=0.8, test_size=0.2)
# asd数据效果好的

def load_images(data_dir, label, target_size=(28, 28)):
    image_list = []
    label_list = []

    print(f"Loading images from directory: {data_dir}")

    for filename in os.listdir(data_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.jpg', '.png', '.jpeg']:
            file_path = os.path.join(data_dir, filename)
            # print(f"Processing file: {file_path}")
            try:
                with Image.open(file_path) as img:
                    img = img.convert('L')
                    img = img.resize(target_size)
                    img_array = np.array(img) / 255.0
                    image_list.append(img_array)
                    label_list.append(label)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    print(f"Loaded {len(image_list)} images.")
    return np.array(image_list), np.array(label_list)


# 设置数据目录
base_dir = '/home/wyd/中医舌诊染苔数据'

# 非染苔图片和标签
non_dyed_dir = os.path.join(base_dir, '非染苔')
non_dyed_images, non_dyed_labels = load_images(non_dyed_dir, 0)  # 标签为0

# 染苔图片和标签
dyed_dir = os.path.join(base_dir, '染苔')
dyed_images, dyed_labels = load_images(dyed_dir, 1)  # 标签为1

# 合并数据集
traindata = np.concatenate((non_dyed_images, dyed_images), axis=0)
traindata = traindata.reshape(traindata.shape[0], 28*28).astype('float64')

trainlabel = np.concatenate((non_dyed_labels, dyed_labels))
trainlabel = trainlabel.reshape(-1, 1)
#
# dataFile = '/home/wyd/spikebls/train.mat'
# data = scio.loadmat(dataFile)
# traindata1 = np.double(data['features'])
# trainlabel1 = np.double(data['classes'])
# # # /home/wyd/spikebls/ASDtest.mat'
# # dataFile = '/home/wyd/spikebls/test.mat'
# # data = scio.loadmat(dataFile)
# # testdata = np.double(data['features'])
# # testlabel = np.double(data['classes'])
#
traindata,testdata ,trainlabel, testlabel=train_test_split(traindata, trainlabel, train_size=0.8, test_size=0.2,random_state=12)
#数据效果不好的ASD
# dataFile = '/home/wyd/spikebls/PSDfeatures.mat'
# data = scio.loadmat(dataFile)
# traindata1 = np.double(data['features'])
# trainlabel1 = np.double(data['classes'])
# traindata,testdata ,trainlabel, testlabel=train_test_split(traindata1, trainlabel1, train_size=0.8, test_size=0.2)
# (traindata,trainlabel), (testdata, testlabel) = fashion_mnist.load_data()
# traindata = traindata.reshape(traindata.shape[0], 28*28).astype('float64')/255
# trainlabel =to_categorical(trainlabel, 10)
# testdata = testdata.reshape(testdata.shape[0], 28*28).astype('float64')/255
# testlabel =to_categorical(testlabel, 10)
N1 = 10  #  # of nodes belong to each window
N2 =10  #  # of windows -------Feature mapping layer
N3 = 7000 #  # of enhancement nodes -----Enhance layer
L = 5    #  # of incremental steps 
M1 = 50  #  # of adding enhance nodes
s = 0.8  #  shrink coefficient
C = 2**-30 # Regularization coefficient

print('-------------------BLS_BASE---------------------------')
BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)

# print('-------------------BLS_ENHANCE------------------------')
# BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
# print('-------------------BLS_FEATURE&ENHANCE----------------')
# M2 = 50  #  # of adding feature mapping nodes
# M3 = 50  #  # of adding enhance nodes
# BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)









'''
teA = list() #Testing ACC 
tet = list() #Testing Time
trA = list() #Training ACC
trt = list() #Training Time
t0 = 0
t2 =[]
t1 = 0
tt1 = 0
tt2 = 0
tt3 = 0
# BLS parameters
s = 0.8  #reduce coefficient
C = 2**(-30) #Regularization coefficient
N1 = 22  #Nodes for each feature mapping layer window 
N2 = 20  #Windows for feature mapping layer
N3 = 540 #Enhancement layer nodes
#  bls-网格搜索
for N1 in range(8,25,2):
    r1 = len(range(8,25,2))
    for N2 in range(10,21,2):
        r2 = len(range(10,21,2))
        for N3 in range(600,701,10):
            r3 = len(range(600,701,10))
            a,b,c,d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3)
            t0 += 1
            if a>t1:
                tt1 = N1
                tt2 = N2
                tt3 = N3
                t1 = a
            teA.append(a)
            tet.append(b)
            trA.append(c)
            trt.append(d)
            print('percent:' ,round(t0/(r1*r2*r3)*100,4),'%','The best result:', t1,'N1:',tt1,'N2:',tt2,'N3:',tt3)
meanTeACC = np.mean(teA)
meanTrTime = np.mean(trt)
maxTeACC = np.max(teA)   
'''
'''
#BLS随机种子搜索
teA = list() #Testing ACC 
tet = list() #Testing Time
trA = list() #Training ACC
trt = list() #Train Time
t0 = 0
t = 0
t2 =[]
t1 = 0
tt1 = 0
tt2 = 0
tt3 = 0
## BLS parameters
s = 0.8 #reduce coefficient
C = 2**(-30) #Regularization coefficient
#N1 = 10  #Nodes for each feature mapping layer window 
#N2 = 10  #Windows for feature mapping layer
#N3 = 500 #Enhancement layer nodes
dataFile = './/dataset//mnist.mat'
data = scio.loadmat(dataFile)
traindata,trainlabel,testdata,testlabel = np.double(data['train_x']/255),2*np.double(data['train_y'])-1,np.double(data['test_x']/255),2*np.double(data['test_y'])-1
u = 45
i = 0
L = 5
M = 50
for N1 in range(10,21,20):
    r1 = len(range(10,21,20))
    for N2 in range(12,21,10):
        r2 = len(range(12,21,10))
        for N3 in range(4000,4001,500):
            r3 = len(range(4000,4001,500))
            for i in range(-28,-27,2):
                r4 = len(range(-28,-27,2))
                C = 2**(i)
#                a,b,c,d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,u)
                a,b,c,d = BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M)
#                t0 += 1
#                if a>t1:
#                    tt1 = N1
#                    tt2 = N2
#                    tt3 = N3
#                    t1 = a
#                    t = u
#                    i1 = i
#                tet.append(b)    
#                teA.append(a)               
#                trA.append(c)
#                trt.append(d)
#                print('NO.',t0,'total:',r1*r2*r3,'ACC:',a*100,'Pars:',N1,',',N2,',',N3,'C',i)
#                print('The best so far:', t1*100,'N1:',tt1,'N2:',tt2,'N3:',tt3,'C:',i1)
                print('working ...')
                print('teACC',teA,'teTime',tet,'trACC',trA,'trTime',trt)
'''
#Grid search for Regularization coefficient 

#
'''
teA = list()
tet = list()
trA = list()
trt = list()
s = 0.8
#C = 2**(-30)
N1 = 10
N2 = 100
N3 = 8000
L = 5
M1 = 20
M2 = 20
M3 = 50
t0 = 0
t1 = 0
t2 = 0
for i in range(-30,-21,5):
    r1 = len(range(-30,-21,5))
    for u in range(10,50,1):
        r2 = len(range(10,50,1))
        C = 2**i
        t0 += 1 
#    a,b,c,d = BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M1)
        a,b,c,d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,u)
        teA.append(a)
        tet.append(b)
        trA.append(c)
        trt.append(d)
        if a > t1:
            t1 = a
            t2 = i
            t = u
        print(t0,'percent:',round(t0/(r1*r2)*100,4),'%','The best result:', t1,'C',t2,'u:',t)
'''     








