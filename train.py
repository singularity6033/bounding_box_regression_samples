# import the necessary packages
from tensorflow.keras.models import load_model
from scipy.io import loadmat
from pyimagesearch import config
from pyimagesearch.iou import compute_iou
from pyimagesearch.ap import compute_ap
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from math import *
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []
# load annotation path
Annotation_Path = config.ANNOTS_PATH
Annotation_Path_List = os.listdir(Annotation_Path)
for filename in Annotation_Path_List:
    for matPath in paths.list_files(os.path.join(Annotation_Path, filename), validExts=".mat"):
        # load the contents of the current mat annotations file
        # print(matPath)
        startY, endY, startX, endX = loadmat(matPath)['box_coord'].flatten().tolist()
        image_detail = 'image' + matPath[-9:-4] + '.jpg'
        # derive the path to the input image, load the image (in
        # OpenCV format), and grab its dimensions
        imagePath = os.path.sep.join([config.IMAGES_PATH, filename, image_detail])
        image = cv2.imread(imagePath)
        (h, w) = image.shape[:2]
        # scale the bounding box coordinates relative to the spatial
        # dimensions of the input image
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h
        # load the image and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        # update our list of data, class labels, bounding boxes, and
        # image paths
        data.append(image)
        labels.append(filename)
        bboxes.append((startX, startY, endX, endY))
        imagePaths.append(imagePath)

# convert the data, class labels, bounding boxes, and image paths to
# NumPy arrays, scaling the input pixel intensities from the range
# [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(lb.classes_[0])
# only there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well
if len(lb.classes_) == 2:
    labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split1 = train_test_split(data, labels, bboxes, imagePaths,
                          test_size=0.20, random_state=42)
# then we divide some for validation (20% of training set)
split2 = train_test_split(split1[0], split1[2], split1[4], split1[6],
                          test_size=0.20, random_state=42)
# unpack the data split
testImages = split1[1]
testLabels = split1[3]
testBBoxes = split1[5]
testPaths = split1[7]
(trainImages, validationImages) = split2[:2]
(trainLabels, validationLabels) = split2[2:4]
(trainBBoxes, validationBBoxes) = split2[4:6]
(trainPaths, validationPaths) = split2[6:]

# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()
# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid",
                 name="bounding_box")(bboxHead)
# construct a second fully-connected layer head, this one to predict
# the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax",
                    name="class_label")(softmaxHead)
# put together our model which accept an input image and then output
# bounding box coordinates and a class label
model = Model(
    inputs=vgg.input,
    outputs=(bboxHead, softmaxHead))
# define a dictionary to set the loss methods -- categorical
# cross-entropy for the class label head and mean absolute error
# for the bounding box head
losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error",
}
# define a dictionary that specifies the weights per loss (both the
# class label and bounding box outputs will receive equal weight)
lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=config.INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())
# construct a dictionary for our target training outputs
trainTargets = {
    "class_label": trainLabels,
    "bounding_box": trainBBoxes
}
# construct a second dictionary, this one for our target testing
# outputs
validationTargets = {
    "class_label": validationLabels,
    "bounding_box": validationBBoxes
}
# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(validationImages, validationTargets),
    batch_size=config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    verbose=1)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")
# serialize the label binarizer to disk
print("[INFO] saving label binarizer...")
f = open(config.LB_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()
# plot the total loss, label loss, and bounding box loss
lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
# loop over the loss names
for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history["val_" + l], label="val_" + l)
    ax[i].legend()
# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plotPath = os.path.sep.join([config.PLOTS_PATH, "losses.png"])
plt.savefig(plotPath)
plt.close()
# create a new figure for the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"],
         label="class_label_train_accuracy")
plt.plot(N, H.history["val_class_label_accuracy"],
         label="val_class_label_accuracy")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
# save the accuracies plot
plotPath = os.path.sep.join([config.PLOTS_PATH, "accs.png"])
plt.savefig(plotPath)
plt.close()

# calculate AP and mAP.
NUM_CLASS = np.size(testLabels, axis=1)
MIN_IOU = [0.5, 0.55, 0.6, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95]
AP = [[] for i in range(NUM_CLASS)]
CORRECT = [[] for i in range(NUM_CLASS)]
Precision = [[] for i in range(NUM_CLASS)]
Recall = [[] for i in range(NUM_CLASS)]
TP_FN = np.zeros(NUM_CLASS)
for min_iou in MIN_IOU:
    for idx, img in enumerate(testImages):
        img = np.expand_dims(img, axis=0)
        # predict the bounding box of the object along with the class.
        i = np.argmax(testLabels[idx])
        TP_FN[i] += 1
        (boxPreds, labelPreds) = model.predict(img)
        iou = compute_iou(boxPreds[0], testBBoxes[0])
        if iou > min_iou:
            CORRECT[i].append(1)
        else:
            CORRECT[i].append(0)
    for j in range(len(CORRECT)):
        for k in range(len(CORRECT[j])):
            Precision[j].append(sum(CORRECT[j][0:k]) / (len(CORRECT[j][0:k]) + 1))
            Recall[j].append(sum(CORRECT[j][0:k]) / TP_FN[j])
        AP[j].append(compute_ap(Precision[j], Recall[j]))
mAP = np.mean(AP, axis=0)

plt.figure()
plt.style.use("ggplot")
plt.title("Class Average Precision")
for i in range(NUM_CLASS):
    plt.plot(MIN_IOU, AP[i])
plt.xlabel("IOU")
plt.ylabel("AP")
plt.legend([lb.classes_[0], lb.classes_[1], lb.classes_[2]], loc="best")
plotPath = os.path.sep.join([config.PLOTS_PATH, "AP.png"])
plt.savefig(plotPath)
plt.close()

plt.figure()
plt.style.use("ggplot")
plt.title("Mean Average Precision")
plt.plot(MIN_IOU, mAP)
plt.xlabel("IOU")
plt.ylabel("mAP")
plotPath = os.path.sep.join([config.PLOTS_PATH, "mAP.png"])
plt.savefig(plotPath)
plt.close()
