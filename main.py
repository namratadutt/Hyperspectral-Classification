# Import libraries
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl 
import scipy.io
from sklearn import metrics
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import BatchNormalization

# Get path to your current directory
basedir = os.getcwd()

# Path to your dataset
filename = basedir + "/hyperSpec.mat"

# Open .mat file with scipy
hyper = scipy.io.loadmat(filename)

# Hyperspectral data
hyper = hyper['hyper']

print(hyper.shape)

# Path to truth dataset
truth_file = basedir + "/muufl_gulfport_campus_1_hsi_220_label.mat"
mat = scipy.io.loadmat(truth_file)
hsi = ((mat['hsi'])[0])[0]

# Ground truth
truth = ((hsi[-2])[0])[-1]
truth = truth[-1]

print(truth.shape)

# Reshape hyperspectral data
hyper = hyper.reshape(325*220, 64)
truth = truth.flatten()

truth = truth - 1
indx, = np.where(truth >= 0)
hyper = hyper[indx]
truth = truth[indx]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(hyper, truth, test_size= 0.3)

np.savez_compressed(basedir +"/train_test_split_hsi.npz", X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)

file = np.load(basedir+"/train_test_split_hsi.npz")
X_train = file['X_train']
X_test = file['X_test']
y_train = file['y_train']
y_test = file['y_test']

X_train[X_train > 1] = 1
X_train[X_train < 0] = 0
X_test[X_test > 1] = 1
X_test[X_test < 0] = 0

# One-hot encoding of labels
y_train = to_categorical(y_train, num_classes = 11, dtype ="int32")
y_test = to_categorical(y_test, num_classes = 11, dtype ="int32")

print(X_train.shape)

model = Sequential()
model.add(Dense(32, input_dim = 64, activation='tanh'))
model.add(BatchNormalization())
model.add(Dense(16, activation='tanh'))
model.add(BatchNormalization())
model.add(Dense(11, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer= 'Adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=128, verbose= 1)
model.save(os.getcwd()+ "/hsi.h5")
model = load_model(os.getcwd()+"/hsi.h5")

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis= -1)
y_test = np.argmax(y_test, axis= -1)

correct = len(y_pred) - np.count_nonzero(y_pred - y_test)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100
   
print("Accuracy: ", acc)

# Compute Confusion matrix
class_names = ['Trees', 'Mostly grass', 'Mixed ground', 'Dirt and sand', 'road', 'water', 'building shadow', 'building', 'sidewalk', 'yellow curb', 'cloth panels']

cm = confusion_matrix(y_test, y_pred, normalize= 'true')
cm = np.round(cm, 3)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot()
plt.show()
