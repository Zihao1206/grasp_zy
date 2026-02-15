import os
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ori_txt_path', default='C:/grasp_static/zoneyung/images', type=str, help='input label path')
parser.add_argument('--txt_path', default='C:/grasp_static/zoneyung', type=str, help='txt label path')
opt = parser.parse_args()
# trainval_percent = 1
train_percent = 0.905
txtfilepath = opt.ori_txt_path
txtsavepath = opt.txt_path
total_xml = os.listdir(txtfilepath)
num = len(total_xml) # 885
print(num)
list_index = range(num)
tv = int(num * train_percent)
# tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
# train = random.sample(trainval, tr)
# file_trainval = open(txtsavepath + '/trainval2.txt', 'w')
# file_val = open(txtsavepath + '/val2.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
# file_val = open(txtsavepath + '/val.txt', 'w')
file_path = "C:/grasp_static/zoneyung/images/"
for i in list_index:
    name = total_xml[i][:-4] + ".jpg" +'\n'
    if i in trainval:
        # file_trainval.write(file_path+name)
        # if i in train:
        file_train.write(file_path+name)
    else:
        file_test.write(file_path+name)
    # else:
    #     file_test.write(file_path+name)
# file_trainval.close()
file_train.close()
file_test.close()
# file_val.close()