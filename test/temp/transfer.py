import numpy as np
from PIL import Image
import os  
path = 'C:/Users/xsy/Desktop/t/fashion_image_label/fashion_test_jpg_10000/'
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg':  # 想要保存的文件格式
                L.append(os.path.join(root, file))  
    return L

def open_imge(file_name):
    im = Image.open(file_name)
    im = np.array(im)
    #print(im.shape)
    return im
def get_label(file_name):
    f = os.path.split(file_name)[1]
    f = f.split('_')[1]
    f = f.split('.')[0]
    return int(f)

files = file_name(path)
numbers = len(files)
H, W = 28, 28
save_features = np.empty((numbers, H, W), dtype=np.float64)
save_labels = np.empty((numbers, ), dtype=np.int64)

for i, f in enumerate(files):
    print(i)
    feature = open_imge(f)
    label = get_label(f)
    save_features[i] = feature
    save_labels[i] = label

np.savez('C:/Users/xsy/Desktop/t/test.npz', **{'features':save_features, 'labels':save_labels})