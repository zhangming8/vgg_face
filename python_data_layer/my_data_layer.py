# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import os
import sys
import cv2
caffe_root='/data/Experiments/caffe'
sys.path.insert(0,caffe_root+'/python')
import caffe

class PythonDataLayer(caffe.Layer):
    def setup(self, bottom,top): 
        # 提取训练集/验证集的 路径和label
        #'/data/msra_lfw/msra/ 64'
        self.data_root, self.batch_size = self.param_str.split(' ')
        self.batch_size = int(self.batch_size)

        self.index = 0
        self.img_list = []

        #print 'self.phase', self.phase
        #if self.phase == 'train':
        if self.phase == 0: #训练集
            for line in open(self.data_root + 'tra_label.txt'): #tra_label.txt：图片路径+标签
                self.img_list.append(line.strip().split()) #img_list所有路径+标签,如['train/57837.png', '5']
        #elif self.phase == 'test':
        elif self.phase == 1: #验证集
            for line in open(self.data_root + 'tra_label.txt'): #split_val_list：图片路径+标签
                self.img_list.append(line.strip().split())
        self.num_img = len(self.img_list) #训练集/验证集 个数
        
    def reshape(self,bottom,top):
        top[0].reshape(self.batch_size, 3, 224, 224)
        top[1].reshape(self.batch_size, 1) #是分类标签只有一列,若为回归标签可能多列
        
    def forward(self,bottom,top):
        #print 'self.num_img', self.phase, self.num_img
        for i in range(self.batch_size):
            if self.phase == 1:
                # sequential sampling
                img_fn, label = self.img_list[self.index%self.num_img]
            elif self.phase == 0:
                # random sampling
                img_fn, label = self.img_list[np.random.randint(self.num_img)]

#            img = cv2.imread(img_fn, 0)
#            img = np.array(img, dtype=np.float32) / 255.0
#            top[0].data[i,0,:,:]=img
            img = Image.open(img_fn) 
            img = np.array(img, dtype=np.float32)
            img = img[:,:,::-1]
            img -= [129.1863,104.7624,93.5940]
            img = img.transpose((2,0,1))
            top[0].data[i,:,:,:]=img
            top[1].data[i,0]=label
            self.index += 1
           
    def backward(self,top,propagate_Down,bottom):
        pass
        