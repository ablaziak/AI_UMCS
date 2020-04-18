# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:52:05 2020

@author: alber
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


from keras import Sequential
from keras.layers import Dense
from keras import optimizers

from termcolor import colored

dataset = pd.read_csv('iris.csv')
dataset.head()

X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

sc = StandardScaler()
Xs = sc.fit_transform(X)

encoder = LabelEncoder()
y1 = encoder.fit_transform(y)

Y = pd.get_dummies(y1).values


modelName = 'model'

classifier = Sequential()
classifier.add(Dense(30, activation = 'relu', kernel_initializer = 'random_normal', input_dim = 4))
classifier.add(Dense(10, activation = 'relu', kernel_initializer = 'random_normal'))
classifier.add(Dense(5, activation = 'relu', kernel_initializer = 'random_normal'))
classifier.add(Dense(3, activation = 'softmax', kernel_initializer = 'random_normal'))

adam = optimizers.Adam(learning_rate = 0.001)
classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics =['acc'])

history = classifier.fit(Xs, Y, batch_size = 1, epochs = 50)


print("-----EVAL MODEL-------")
eval_model=classifier.evaluate(Xs, Y)
print(eval_model)

print("-------------")
y_pred = classifier.predict(Xs)
y_pred2 = (y_pred>0.7)

for i in range(0, len(y_pred2)):
    if (y_pred2[i][0]):
        print(colored('setosa', 'red'))
    elif(y_pred2[i][1]):
        print(colored('versicolor', 'blue'))
    else:
        print(colored('virginica', 'green'))
