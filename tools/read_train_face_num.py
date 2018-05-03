# -*- coding: utf-8 -*-

#%% 提取出每一类图片的 标签与个数
label_landmark_path = '/data/msra_lfw/msra/msra_lb_lmk'
label_landmark = open(label_landmark_path).readlines() #old label and landmark file

every_class_num = {}
every_class_num[0] = 0
last_label = 0
for line in label_landmark:
    temp = line.split(' ')
    img_path, label, landmark = temp[0].split('/'), int(temp[1]), temp[2:]
    
    if label == last_label:
        if label not in every_class_num:
            every_class_num[label] = 0
        every_class_num[label] += 1 
    last_label = label
print('class_num is:',len(every_class_num))

with open('./every_class_num.txt', 'w') as f:
    for i in every_class_num:
        f.write(str(i) + ' ' + str(every_class_num[i]+1))
        f.write('\n')
        

#%% 观察每一类样本中大于 xx 个图片的个数        
more_than_xx_label = []        
xx = 80
f = open('./every_class_num.txt')
data = f.readlines()
with open('./more_than_xx_label.txt','w') as f2:
    for line in data:
        label, num = line.split(' ')
        label, num = int(label), int(num)
        if num > xx:
            more_than_xx_label.append(label)
            f2.write(str(label))
            f2.write('\n')
f.close
print('more than %d picture label is:'% xx,len(more_than_xx_label))