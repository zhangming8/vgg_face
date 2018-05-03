# -*- coding: utf-8 -*-
# 用于显示网络结构
import numpy as np
import matplotlib.pyplot as plt
import sys
caffe_root = '/data/Experiments/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe
caffe.set_mode_cpu()

model_def='./deploy_VGG_FACE.prototxt'
model_weights='./VGG_FACE.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)
print('The structure of the NET is shown as following:')
for layer_name, blob in net.blobs.iteritems():
    print(layer_name+'\t'+str(blob.data.shape))