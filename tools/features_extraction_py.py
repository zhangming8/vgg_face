# -*- coding: utf-8 -*-
#https://www.cnblogs.com/louyihang-loves-baiyan/p/5078746.html
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import struct
import sys,cv2
caffe_root = '/data/Experiments/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe

# 特征保存的路径
save_dir = '/data/zhangming/vgg_face/features/features.ffff'
# 运行模型的prototxt
deployPrototxt = '/data/zhangming/vgg_face/deploy_train_val_vgg16_no_lmbd.prototxt'
# 相应载入的modelfile
modelFile = '/data/zhangming/vgg_face/snapshot/vgg_face_iter_100000.caffemodel'
# meanfile 也可以用自己生成的
#meanFile = 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
# 需要提取的图像列表
imageListFile = '/data/zhangming/vgg_face/features/file_list.txt'
#imageBasePath = '/home/chenjie/DataSet/500CarCNNRetrieve/500CarFaceOrig'
gpuID = 0

# 初始化函数的相关操作
def initilize():
    print('initilize ... ')

    sys.path.insert(0, caffe_root + 'python')
#    caffe.set_mode_gpu()
#    caffe.set_device(gpuID)
    net = caffe.Net(deployPrototxt, modelFile,caffe.TEST)
    return net  
# 提取特征并保存为相应地文件
def extractFeature(imageList, net):
    # 对输入数据做相应地调整如通道、尺寸等等
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
#    transformer.set_mean('data', np.load(caffe_root + meanFile).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))  
    # set net to batch size of 1 如果图片较多就设置合适的batchsize 
    net.blobs['data'].reshape(1,3,224,224)      #这里根据需要设定，如果网络中不一致，需要调整
    num=0
    all_fe = np.array([])
    for imagefile in imageList:
 #       imagefile_abs = os.path.join(imageBasePath, imagefile)
        imagefile_abs = imagefile
 #       p1rint imagefile_abs
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagefile_abs))
        out = net.forward()
        num +=1
        print(num,imagefile_abs)
        #for x in xrange(0, net.blobs['fc7'].data.shape[0]):
         #   for y in xrange(0, net.blobs['fc7'].data.shape[1]):
           #     f.write(struct.pack('f', net.blobs['fc7'].data[x,y])) 
        features = net.blobs['fc7'].data  # 提取卷积层 5 的特征
        all_fe = np.append(all_fe, features)
    all_fe = np.reshape(all_fe,[-1,4096])        
    np.savetxt('fc7_features.txt', all_fe) # 将特征存储到本文文件中

# 读取文件列表
def readImageList(imageListFile):
    imageList = []
    with open(imageListFile,'r') as fi:
        while(True):
            line = fi.readline().strip().split()# every line is a image file name
            if not line:
                break
            imageList.append(line[0]) 
    print('image number:', len(imageList))
    return imageList

if __name__ == "__main__":
    net = initilize()
    imageList = readImageList(imageListFile) 
    extractFeature(imageList, net)
