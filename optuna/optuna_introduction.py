# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:45:13 2020

@author: alber
"""
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras import Sequential
from keras.layers import Dense
from keras import optimizers

from termcolor import colored

import matplotlib.pyplot as plt

import optuna
    
dataset = pd.read_csv('D:\SpyderFiles\Irys\iris.csv')

sc = StandardScaler()
encoder = LabelEncoder()
X = sc.fit_transform(dataset.iloc[:,0:4].values)
y = encoder.fit_transform(dataset.iloc[:,4].values)
Y = pd.get_dummies(y).values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


def objective(trial):
    first_layer = trial.suggest_int('first_layer', 20, 40)
    # learning_rate = trial.suggest_uniform('learing_rate', 0.0001, 0.0005)
    epochs = trial.suggest_int('epochs', 50, 150)
    optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'adadelta', 'adagrad', 'nadam', 'rmsprop'])
    
    print('\n\n#',trial.number,'\nCurrent params: first layer = ',first_layer,
          '; epochs = ', epochs, '; optimizer: ' , optimizer, '\n', sep = '')
    
    classifier = Sequential()
    classifier.add(Dense(first_layer, activation = 'relu', kernel_initializer = 'random_normal', input_dim = 4))
    classifier.add(Dense(10, activation = 'relu', kernel_initializer = 'random_normal'))
    classifier.add(Dense(5, activation = 'relu', kernel_initializer = 'random_normal'))
    classifier.add(Dense(3, activation = 'softmax', kernel_initializer = 'random_normal'))
    
    classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['acc'])
    
    history = classifier.fit(X_train, Y_train, batch_size = 4, epochs = epochs, validation_data = (X_test, Y_test))
    
    error = history.history['loss'][-1] #loss/val_loss/mse_loss
    
    return error

study = optuna.create_study()
study.optimize(objective, n_trials=2)
print('\n', study.best_params)

best = study.best_params
classifier = Sequential()
classifier.add(Dense(best.get('first_layer'), activation = 'relu', kernel_initializer = 'random_normal', input_dim = 4))
classifier.add(Dense(10, activation = 'relu', kernel_initializer = 'random_normal'))
classifier.add(Dense(5, activation = 'relu', kernel_initializer = 'random_normal'))
classifier.add(Dense(3, activation = 'softmax', kernel_initializer = 'random_normal'))

classifier.compile(optimizer = best.get('optimizer'), loss = 'categorical_crossentropy', metrics = ['acc'])

history = classifier.fit(X_train, Y_train, batch_size = 4, epochs = best.get('epochs'), validation_data = (X_test, Y_test))

print("-----EVAL MODEL-------")
eval_model=classifier.evaluate(X_train, Y_train)
print(eval_model)

print("-------------")
y_pred = classifier.predict(X_test)
y_pred2 = (y_pred>0.7)

print("Y_true  |  Y_pred")
for i in range(0, len(y_pred2)):
    if (Y_test[i][0]):
        print(colored('setosa', 'red'), end=' | ')
    elif(Y_test[i][1]):
        print(colored('versicolor', 'blue'), end=' | ')
    else:
        print(colored('virginica', 'green'), end=' | ')
    
    if (y_pred2[i][0]):
        print(colored('setosa', 'red'))
    elif(y_pred2[i][1]):
        print(colored('versicolor', 'blue'))
    else:
        print(colored('virginica', 'green'))
        

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label = 'Train Accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'b', label = 'Train Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


