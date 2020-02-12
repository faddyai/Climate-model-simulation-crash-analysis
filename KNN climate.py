import pandas as pd
import sklearn as sk
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing, cross_decomposition, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

#x_train, y_train = load_svmlight_file("/home/fubunutu/PycharmProjects/midterm/newpop.csv")

df = pd.read_csv("/home/fubunutu/PycharmProjects/midterm/newpop2.csv")

#create array, x values all columns except outcome. Y values = outcome column boolean
x = np.array(df.drop(['outcome'], 1))
y = np.array(df['outcome'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15)
clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)

# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores
cv_scores = []

#15-fold cross validation comparing accuracy of k values

for k in neighbors:
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf, x_train, y_train, cv=15, scoring='accuracy')
    cv_scores.append(scores.mean())

# misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
acclist = []
loops = 10000
loopstr = str(loops)
# i = 1
for i in range(loops):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15)
    clf = KNeighborsClassifier(n_neighbors=optimal_k)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    accstr = str("%.3f" % accuracy)
    # np.append(acclist, accuracy)
    acclist.append(accuracy)
    opt = str(optimal_k)
    # i += 1

# print('The accuracy of the model on the test dataset is ' + accstr + '% with k=' + opt)

y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
# Show confusion matrix in a separate window

plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

model = ExtraTreesClassifier(n_estimators=1000)
model.fit(x, y)
labels = df.head(0)
importance =np.array(model.feature_importances_)



print("The importance (0-1) of each feature with regard to the outcome ")
print(importance)
print("The average accuracy from " + loopstr + " simulations is:")
print(sum(acclist)/len(acclist))