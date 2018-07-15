import scipy
import sklearn
import sklearn.cluster
import pickle
from enum import Enum

from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

class Objects(Enum):
    banana=1
    bulldozer=2
    chair=3
    eyeglasses=4
    flashlight=5
    foot=6
    hand=7
    harp=8
    hat=9
    keyboard=10
    laptop=11
    nose=12
    parrot=13
    penguin=14
    pig=15
    skyscraper=16
    snowman=17
    spider=18
    trombone=19
    violin=20

def create_train():
    lengths=[]
    filename="/home/cse/btech/cs1150247/Assignment4/Dataset/train/"+Objects(1).name+ ".npy"
    obj_np=np.load(filename)
    obj_np=np.array(obj_np/255)
    lengths.append(len(obj_np))
    for i in range(1,20):
        filename="/home/cse/btech/cs1150247/Assignment4/Dataset/train/"+Objects(i+1).name+ ".npy"
        temp=np.load(filename)
        temp=np.array(temp/255)
        lengths.append(len(temp))
        obj_np=np.vstack((obj_np,temp))


#     print(lengths)
    return obj_np, lengths
        
X_train, lengths=create_train()
X_train= StandardScaler().fit_transform(X_train)

X_conv_train=X_train.reshape(100000, 28,28,1)

import csv
# print(X_train)
def csv_writer(test_answers):
    with open('/home/cse/btech/cs1150247/scratch/answers1.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(["ID", "CATEGORY"])
        for i in range(len(test_answers)):
            mywriter.writerow([i, Objects(test_answers[i]+1).name])

def createYtrain():
    Y_train=[]
    for i in range(20):
        for j in range(lengths[i]):
            temp=np.zeros((1,20))
            temp[0][i]+=1
            Y_train.append(temp[0])
    return np.array(Y_train)

def createY():
    Y_train=[]
    for i in range(20):
        for j in range(lengths[i]):
            Y_train.append(i)
    return np.array(Y_train)

Y_train=createYtrain()
print(Y_train)

def create_test():
    filename="/home/cse/btech/cs1150247/Assignment4/Dataset/test/test.npy"
    obj_np=np.load(filename)
    return obj_np

#     print(lengths)
X_test=create_test()
X_test= StandardScaler().fit_transform(X_test)

X_tst_conv=X_test.reshape(100000,28,28,1)

from sklearn.model_selection import train_test_split 
for i in range(1):
    trainX, testX, trainY, testY = train_test_split(X_conv_train, Y_train, test_size=0.0, shuffle=True)

from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D

model=Sequential()
model.add(Conv2D(64, kernel_size=(4, 4),activation='relu',input_shape=(28,28,1), padding='same'))
model.add(Conv2D(128, kernel_size=(4, 4),activation='relu',input_shape=(28,28,1), padding='same'))
model.add(AveragePooling2D(pool_size=(3, 3)))
model.add(Conv2D(128, kernel_size=(4, 4),activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit(trainX, trainY, epochs=25, batch_size=100)
test_out=model.predict(X_tst_conv)
# print(test_out)
labels=np.argmax(test_out, axis=1)
print(len(labels))

csv_writer(labels)
model.save('/home/cse/btech/cs1150247/scratch/newmodel1.h5')









