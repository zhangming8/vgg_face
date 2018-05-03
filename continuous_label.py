# -*- coding: utf-8 -*-
#%% 使得tra内的标签连续从0-9999 共10000类
label_dic = {}
num = 1
label = 0
with open('tra_continuous_label_224x224.txt','w') as f:
    with open('tra_label_224x224.txt') as tra_label:
        for line in tra_label:
            path, old_label = line.split(' ')
            f.write(path + ' ' + str(label))
            f.write('\n')
            label_dic[int(old_label)] = str(label)
            if num  == 80: ##### 80为训练集每类有80个图片
                label += 1
                num = 0
            num += 1
                  
#%% 让验证集的标签对应更换
with open('val_continuous_label_224x224.txt','w') as f2:
    with open('val_label_224x224.txt') as val_label:
        for line in val_label:
            path, old_label = line.split(' ')
            f2.write(path +' '+ label_dic[int(old_label)])
            f2.write('\n')