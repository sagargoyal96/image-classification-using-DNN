import scipy
import sklearn
import sklearn.cluster
import pickle
from enum import Enum

from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv

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
    filename="/home/cse/btech/cs1150251/scratch/train/"+Objects(1).name+ ".npy"
    obj_np=np.load(filename)
    obj_np=np.array(obj_np/255)
    lengths.append(len(obj_np))
    for i in range(1,20):
        filename="/home/cse/btech/cs1150251/scratch/train/"+Objects(i+1).name+ ".npy"
        temp=np.load(filename)
        temp=np.array(temp/255)
        lengths.append(len(temp))
        obj_np=np.vstack((obj_np,temp))


#     print(lengths)
    return obj_np, lengths



# def csv_writer(test_answers):
#     with open('/home/cse/btech/cs1150251/scratch/answers.csv', 'w') as csvfile:
#         mywriter = csv.writer(csvfile, delimiter=',')
#         mywriter.writerow(["ID", "CATEGORY"])
#         for i in range(len(test_answers)):
#             mywriter.writerow([i, Objects(test_answers[i]+1).name])


def csv_reader():
    main_arr=np.zeros((100000,20))
    for i in range(6):
        with open((i+1).str()+".csv", 'r') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',')
            for row in myreader:
                print(row)
                break


csv_reader()




















