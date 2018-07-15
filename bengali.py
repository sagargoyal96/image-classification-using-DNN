# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle
# fix random seed for reproducibility
np.random.seed(3)
# load pima indians dataset
def read_npy(directory):
    iterr=0
    label = []
    files = os.listdir(directory)
    for file_name in files:
        numpy_file = np.load(directory + file_name)
        if iterr==0:
            data = numpy_file
        else:
            data = np.concatenate((data,numpy_file))
        label.append(file_name[:-4])
        iterr+=1
    return data,label

data,label = read_npy("/home/cse/btech/cs1150251/scratch/train/")
t_data,_ = read_npy("/home/cse/btech/cs1150251/scratch/test/")
print(data.shape)
#print(t_data.shape)
print(label)
labels = []
tr_data = []
data = StandardScaler().fit_transform(data)
t_data = StandardScaler().fit_transform(t_data)
print(data[0])
data = data.reshape(100000,28,28,1)
print(data[0])
t_data = t_data.reshape(100000,28,28,1)
for i in range(0,len(data)):
	st = np.zeros(20)
	st[int(i/5000)] = 1
	labels.append(st)
labels = np.array(labels)

# split into input (X) and output (Y) variables
# create model
print(data.shape)
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=[28,28,1]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(data,labels, epochs=20,batch_size=2000)

model_json = model.to_json()
with open("/home/cse/btech/cs1150251/scratch/model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/cse/btech/cs1150251/scratch/model2.h5")
print("Saved model to disk")
# evaluate the model
predict = []
predictions = model.predict(t_data)
print(predictions[0])
for x in predictions:
    y_pre = np.argmax(x)
    #print(y_pre)
    predict.append(y_pre)

with open('/home/cse/btech/cs1150251/scratch/nn1.csv','w') as file:
    file.write("ID,CATEGORY")
    file.write("\n")
    for i in range(len(predict)):
        file.write(str(i)+","+str(label[int(predict[i])]))
        file.write('\n')
# scores = model.evaluate(data,labels)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))