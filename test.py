# import imutils
# from scipy.io import loadmat
from pyimagesearch import config
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# from imutils import paths
import matplotlib.pyplot as plt
# import numpy as np
# import pickle
# import cv2
import os
#
# # imagePath = 'output/test_paths.txt'
# # imagePaths = open(imagePath).read().strip().split("\n")
#
# # load annotation path
# # Annotation_Path = config.ANNOTS_PATH
# # Annotation_Path_List = os.listdir(Annotation_Path)
# # for filename in Annotation_Path_List:
# #     for matPath in paths.list_files(os.path.join(Annotation_Path, filename), validExts=".mat"):
# #         # load the contents of the current mat annotations file
# #         # print(matPath)
# #         startX, startY, endX, endY = loadmat(matPath)['box_coord'].flatten().tolist()
# #         image_detail = 'image'+matPath[-9:-4]+'.jpg'
# #         # derive the path to the input image, load the image (in
# #         # OpenCV format), and grab its dimensions
# #         imagePath = os.path.sep.join([config.IMAGES_PATH, filename, image_detail])
# #         image = cv2.imread(imagePath)
# #         print(image)
#
#
# startY, endY, startX, endX = loadmat('dataset/annotations/airplanes/annotation_0002.mat')[
#     'box_coord'].flatten().tolist()
# image = cv2.imread('dataset/images/airplanes/image_0002.jpg')
# # image = imutils.resize(image, width=600)
# (h, w) = image.shape[:2]
# # startX = int(startX * w)
# # startY = int(startY * h)
# # endX = int(endX * w)
# # endY = int(endY * h)
# # draw the predicted bounding box and class label on the image
# y = startY - 10 if startY - 10 > 10 else startY + 10
# cv2.rectangle(image, (startX, startY), (endX, endY),
#               (0, 255, 0), 2)
# # show the output image
# cv2.imshow("Output", image)
# cv2.waitKey(0)
import numpy as np
from matplotlib import pyplot as plt

a = [0.1, 0.2, 0.3, 0.4]
AP = [[0.8, 0.9, 0.7,0.8], [0.9, 0.5, 0.3,0.8], [0.8, 0.3, 0.6,0.8]]
mAP = np.mean(AP, axis=0)
print(mAP)
# plt.figure()
# plt.style.use("ggplot").
# plt.title("Class Average Precision")
# plt.plot(a[0], AP[0])
# plt.xlabel("IOU")
# plt.ylabel("AP")
# plt.legend(['1', '2', '3'], loc="best")
# plt.show()
# # plotPath = os.path.sep.join([config.PLOTS_PATH, "AP.png"])
# # plt.savefig(plotPath)
#
# plt.close()
# plt.style.use("ggplot")
# plt.title("Mean Average Precision")
# plt.plot(a, mAP)
# plt.xlabel("IOU")
# plt.ylabel("mAP")
# # plotPath = os.path.sep.join([config.PLOTS_PATH, "mAP.png"])
# # plt.savefig(plotPath)
