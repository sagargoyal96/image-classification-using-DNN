import scipy
import sklearn
import sklearn.cluster
import pickle
from enum import Enum

from sklearn.decomposition import PCA
import numpy as np

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
    # filename="/home/cse/btech/cs1150254/scratch/train/"+Objects(1).name+ ".npy"
    filename="train/"+Objects(1).name+ ".npy"
    obj_np=np.load(filename)
    obj_np=np.array(obj_np/255)
    lengths.append(len(obj_np))
    for i in range(1,20):
        # filename="/home/cse/btech/cs1150254/scratch/train/"+Objects(i+1).name+ ".npy"
        filename="train/"+Objects(i+1).name+ ".npy"
        temp=np.load(filename)
        temp=np.array(temp/255)
        lengths.append(len(temp))
        obj_np=np.vstack((obj_np,temp))

    return obj_np, lengths

def create_test():
    lengths=[]
    # filename="/home/cse/btech/cs1150254/scratch/train/"+Objects(1).name+ ".npy"
    filename="test/test.npy"
    obj_np=np.load(filename)
    obj_np=np.array(obj_np/255)
#     print(lengths)
    return obj_np
        
X_train, lengths=create_train()
Y_train=create_test()

import csv
# print(X_train)
def csv_writer(test_answers):
    with open('ss.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(["ID", "CATEGORY"])
        for i in range(len(test_answers)):
            mywriter.writerow([i, Objects(test_answers[i]+1).name])
            

from sklearn.preprocessing import StandardScaler
X_train= StandardScaler().fit_transform(X_train)

def createYtrain():
    Y_train=[]
    for i in range(20):
        for j in range(lengths[i]):
            Y_train.append(i)
    return np.array(Y_train)

def createYtest():
    Y_train=[]
    for i in range(20):
        for j in range(lengths[i]):
            Y_train.append(i)
    return np.array(Y_train)

Y_train=createYtrain()
# print(len(Y_train))
pca = PCA(n_components=50)
newX_train=pca.fit_transform(X_train)

from sklearn.svm import SVC


from sklearn.model_selection import train_test_split



exp_arr=[1]
accuracy_arr=[]
for item in exp_arr:
    max_acc=0
    mysvm=SVC(C=item, kernel='linear', decision_function_shape='ovo')
    for i in range(1):
        print("yessss")
        svm_trainX, svm_testX, svm_trainY, svm_testY = train_test_split(newX_train, Y_train, test_size=0.0, shuffle=True)
        mysvm.fit(svm_trainX, svm_trainY)
        acc=mysvm.score(svm_trainX, svm_trainY)
        max_acc+=acc
    accuracy_arr.append(max_acc)
    print(max_acc)

print(accuracy_arr)

# with open('/home/cse/btech/cs1150251/scratch/svm_sagar.csv','w') as file:
#     # file.write("\n")
#     for i in accuracy_arr:
#         file.write(i)
#         file.write('\n')



# print(max_acc)

















