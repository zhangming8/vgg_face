# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
caffe_root = '/data/Experiments/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe
caffe.set_mode_gpu()

model_def='./deploy_VGG_FACE.prototxt'#你自己的路径
model_weights='./VGG_FACE.caffemodel'#你自己的路径
labels_filename='./names.txt'#你自己的路径

mean_data=np.array([129.1863,104.7624,93.5940])
plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = caffe.Net(model_def, model_weights, caffe.TEST)
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data',mean_data)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
# net.blobs['data'].reshape(1,3,227,227)

image = caffe.io.load_image('./ak.png')
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image

labels=np.loadtxt(labels_filename,str,delimiter='\n')
output = net.forward()
prob = output['prob'][0]

index1=prob.argsort()[-1]
index2=prob.argsort()[-2]
index3=prob.argsort()[-3]
index4=prob.argsort()[-4]
index5=prob.argsort()[-5]

print(index1,'--',labels[index1],'--',prob[index1])
print(index2,'--',labels[index2],'--',prob[index2])
print(index3,'--',labels[index3],'--',prob[index3])
print(index4,'--',labels[index4],'--',prob[index4])
print(index5,'--',labels[index5],'--',prob[index5])
# print output_prob
# print labels[output_prob.argmax()]