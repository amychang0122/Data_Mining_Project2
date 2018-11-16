import numpy as np
import matplotlib.pyplot as plt
import graphviz
import re
import csv   # 導入csv包
import random
import pandas as pd
import subprocess
from sklearn import cross_validation, ensemble, preprocessing, metrics, svm, neighbors, tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

'''
# 產生資料集
with open('Data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for n in range(0,1000) :
        arr = []
        for i in range(0,11) :
            if i == 0 :
                arr.append(random.randint(1600, 2800)/10)
            elif i == 1 :
                arr.append(random.randint(10, 800)/10)
            elif i == 2 :
                arr.append(random.randint(0, 100)/10)
            elif i == 3 :
                arr.append(random.randint(0, 10800)/10)
            elif i == 4 : 
                arr.append(random.randint(15, 40)/10)
            elif i == 5 :
                arr.append(random.randint(50, 300)/10)
            elif i == 6 :
                arr.append(random.randint(20, 100)/10)
            elif i == 7 :
                arr.append(random.randint(50, 300)/10)
            elif i == 8 :
                arr.append(random.randint(1, 150)/10)
            elif i == 9 :
                arr.append(random.randint(1400, 2000)/10)
            else :
                arr.append(random.randint(1, 70)/10)
        writer.writerow(arr)

# 根據規則產生結果
csv_file=csv.reader(open("Data.csv",'r'))
with open('result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    #for stu in csv_file:
    i=0
    for each in csv_file :
        if float(each[2]) < 0.5 :
            writer.writerow('1')
        elif float(each[2]) > 4 :
            writer.writerow('0')
        elif float(each[0])>224 and float(each[1])<35 and float(each[2])<2 and float(each[3])<30 :
            writer.writerow('1')
        elif float(each[8])>=10 and float(each[9])>160 :
            writer.writerow('1')
        elif float(each[5]) > float(each[6])*2.8 :
            writer.writerow('1')
        elif float(each[4]) > 2.6 :
            if float(each[7]) <= 20 and float(each[5]) > 20 :
                writer.writerow('1')
            elif float(each[7])>20 and float(each[9])>175 :
                writer.writerow('1')
            else :
                writer.writerow('0')
        else :
            writer.writerow('0')
'''

csv_file=csv.reader(open("Data.csv",'r'))
dataset = []
for each in csv_file :
    dataset.append(each)

csv_file = csv.reader(open("result.csv", 'r'))
result = []
for each in csv_file :
    result.extend(each)

total_num = 1000
training_rate = 0.7

#訓練集
train_data = dataset[0:(int)(total_num*training_rate)]
train_target = result[0:(int)(total_num*training_rate)]
test_data =dataset[(int)(total_num*(1-training_rate)):total_num]
test_target = result[(int)(total_num*(1-training_rate)):total_num]


## Decision Tree
#####################################################################
clf = DecisionTreeClassifier()
clf.fit(train_data, train_target)
predict_target = clf.predict(test_data)

print("\n\nMatch = ", sum(predict_target == test_target), "/ 700")
print("Confusion_matrix = \n", metrics.confusion_matrix(test_target, predict_target))
print(metrics.classification_report(test_target, predict_target))


## Random Forest
#####################################################################
rfc = RandomForestClassifier()
rfc.fit(train_data, train_target)
rfc_predict = rfc.predict(test_data)

print("\n\nMatch = ", sum(rfc_predict == test_target), "/ 700")
print("Confusion_matrix = \n", metrics.confusion_matrix(test_target, rfc_predict))
print(metrics.classification_report(test_target, rfc_predict))


## SVC
#####################################################################
svc = svm.SVC()  # class   
svc.fit(train_data, train_target)  # training the svc model  
svc_predict = svc.predict(test_data) # predict the target of testing samples  

print("\n\nMatch = ", sum(svc_predict == test_target), "/ 700")
print("Confusion_matrix = \n", metrics.confusion_matrix(test_target, svc_predict))
print(metrics.classification_report(test_target, svc_predict))


## KNN
#####################################################################
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(train_data, train_target)
knn_predict = knn.predict(test_data)

print("\n\nMatch = ", sum(knn_predict == test_target), "/ 700")
print("Confusion_matrix = \n", metrics.confusion_matrix(test_target, knn_predict))
print(metrics.classification_report(test_target, knn_predict))


## 輸出圖形
#####################################################################
dot_data = tree.export_graphviz(clf, out_file="tree.dot",filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data)
command = ["dot", "-T", "png", "tree.dot", "-o", "tree.png"]
try :
    subprocess.check_call(command)
except :
    print("What")
#graph.render("QQQ")
