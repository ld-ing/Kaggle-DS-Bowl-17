
import numpy as np
from keras.applications.resnet50 import ResNet50
import dicom
import cv2
import glob
import os


"""Extract feature using pre-trained ResNet-50.

Input:
    path to the data folders (stage1/stage2)

Output:
    .npy files (feature vectors) for each id, saved to the path
    
"""


def load_extractor():
    network = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None)
    return network


def load_3d_data(path):
    # credit: https://www.kaggle.com/mumech/loading-and-processing-the-sample-images
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


def convert_rgb(path):
    CTs = load_3d_data(path)
    CTs[CTs == -2000] = 0  # -2000 is a flag for missing value
    RGBs = []
    for i in range(0, CTs.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = CTs[i + j]
            img = 255.0 / np.amax(img) * img
            img = np.array(cv2.resize(img, (224, 224)))
            tmp.append(img)
        tmp = np.array(tmp).reshape(224, 224, 3)
        RGBs.append(tmp)
    return np.array(RGBs)


def extract_features(path):
    net = load_extractor()
    i = 1
    for folder in glob.glob(path + '/*'):
        if not (os.path.isfile(folder)):
            if not (os.path.isfile(folder + '.npy')):
                img = convert_rgb(folder)
                feats = net.predict(img, batch_size=img.shape[0], verbose=1)
                np.save(folder, feats)
                print 'Completed:', i
            i += 1


if __name__ == '__main__':
    extract_features('data/stage1')
#    extract_features('data/stage2')
