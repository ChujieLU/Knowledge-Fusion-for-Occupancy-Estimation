import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sklearn.metrics as skm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D,Dropout,concatenate,average
from keras.models import Model
from tensorflow.keras.layers import LSTM, Dense,Bidirectional,Input
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.utils import to_categorical

np.random.seed(0)
tf.random.set_seed(0)

df= pd.read_csv("FinalData_Hand.csv")

#Data preparation
X=df[["Temp",'CO2','VOC','Electricity']]
F=df[['CO2_15','Temp_15','Tempfd','Tempfs','VOC_15','Time15',
       'Powerfd','Win15','Cooling15','Time15']]
y1=df['Presence_15']
y2=df['Number_15']

trainX_raw=X[0:17610] # Raw data
testX_raw=X[17610:]
trainF_raw=F[0:1174] # Handcraft features
trainy1_raw=y1[0:1174]
trainy2_raw=y2[0:1174]
testF_raw=F[1174:1434]
testy1_raw=y1[1174:1434]
testy2_raw=y2[1174:1434]

#Data Preparation
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scalerF = preprocessing.MinMaxScaler()
trainX_nor=min_max_scaler.fit_transform(trainX_raw)
testX_nor=min_max_scaler.transform(testX_raw)
trainF=min_max_scalerF.fit_transform(trainF_raw)
testF=min_max_scalerF.transform(testF_raw)

trainX=np.array(np.split(trainX_nor, len(trainX_nor)/15)) # 15mins as interval
testX=np.array(np.split(testX_nor, len(testX_nor)/15))

#One-hot
trainy1=to_categorical(num_classes=2,y=trainy1_raw)
trainy2=to_categorical(num_classes=4,y=trainy2_raw)
testy1=to_categorical(num_classes=2,y=testy1_raw)
testy2=to_categorical(num_classes=4,y=testy2_raw)

n_steps, n_length, n_features=1,15,4
trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))


def Detection(): # Occupancy Detection
    inputs = Input(shape=(None, 15, 4))
    handfea = Input(shape=(10,))
    Densehand = Dense(100, activation='relu')(handfea)
    densehand = Dense(10, activation='relu')(Densehand)

    CNN1 = TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu'),
                           input_shape=(None, 15, 4))(inputs)
    #    CNN2=TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'))(CNN1)
    Max = TimeDistributed(MaxPooling1D(pool_size=2))(CNN1)
    Flat = TimeDistributed(Flatten())(Max)
    BiLSTM1 = Bidirectional(LSTM(32, return_sequences=0))(Flat)
    #    BiLSTM2=Bidirectional(LSTM(50))(BiLSTM1)
    drop1 = Dropout(0.5)(BiLSTM1)
    Dense1 = Dense(10, activation='relu')(drop1)
    #    Dense2=Dense(10, activation='relu')(Dense1)
    mymerge = average([Dense1, densehand])
    drop2 = Dropout(0.5)(mymerge)
    output = Dense(2, activation='softmax')(drop2)
    model = Model([inputs, handfea], output)
    return model


def Estimation(): # Occupancy Estimation
    inputs = Input(shape=(None, 15, 4))
    handfea = Input(shape=(10,))
    Densehand = Dense(100, activation='relu')(handfea)
    densehand = Dense(10, activation='relu')(Densehand)

    CNN1 = TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu'),
                           input_shape=(None, 15, 4))(inputs)
    #    CNN2=TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'))(CNN1)
    Max = TimeDistributed(MaxPooling1D(pool_size=2))(CNN1)
    Flat = TimeDistributed(Flatten())(Max)
    BiLSTM1 = Bidirectional(LSTM(32, return_sequences=0))(Flat)
    #    BiLSTM2=Bidirectional(LSTM(50))(BiLSTM1)
    drop1 = Dropout(0.5)(BiLSTM1)
    Dense1 = Dense(10, activation='relu')(drop1)
    #    Dense2=Dense(10, activation='relu')(Dense1)
    mymerge = average([Dense1, densehand]) #Merging!
    drop2 = Dropout(0.5)(mymerge)
    output = Dense(4, activation='softmax')(drop2)
    model = Model([inputs, handfea], output)
    return model

model=Detection()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
batch_size=64
epoch_num=200
history = model.fit([trainX,trainF], trainy1, batch_size=batch_size, epochs=epoch_num, verbose=2)
print('\n',model.summary())

model_e=Estimation()
model_e.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
print('\n',model_e.summary())
history = model_e.fit([trainX,trainF], trainy2, batch_size=batch_size, epochs=epoch_num, verbose=2)

plt.plot(history.history['loss'])
hty=history.history['loss']

loss,accuracy=model.evaluate([testX,testF], testy1)
loss_e,accuracy_e=model_e.evaluate([testX,testF], testy2)

#train
train_pre_raw=model.predict([trainX,trainF])
train_pre=np.argmax(train_pre_raw,axis=1)
train_pre_num_raw=model.predict([trainX,trainF])
train_pre_num=np.argmax(train_pre_num_raw,axis=1)

#test
pre_raw=model.predict([testX,testF])
pre=np.argmax(pre_raw,axis=1)
pre_num_raw=model_e.predict([testX,testF])
pre_num=np.argmax(pre_num_raw,axis=1)

#Detection Results
figsize=(12,3)
fig = plt.figure(figsize=figsize)
plt.plot(testy1_raw.values,linestyle=':',label="Groundtruth")
plt.plot(pre,label='CDBLSTM')
plt.legend()
plt.show()

#Estimation Results
figsize=(12,3)
fig = plt.figure(figsize=figsize)
plt.plot(testy2_raw.values,linestyle=':',label="Groundtruth")
plt.plot(pre_num,label='CDBLSTM')
plt.legend()
plt.show()

#Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score

#Training performance
print("Detectionaccuracy_score:", accuracy_score(trainy1_raw, train_pre))
print("Estimationaccuracy_score:", accuracy_score(trainy2_raw, train_pre_num))
tmse_d=metrics.mean_squared_error(trainy1_raw, train_pre)
tmse_e=metrics.mean_squared_error(trainy2_raw, train_pre_num)
tmae_d=metrics.mean_absolute_error(trainy1_raw, train_pre)
tmae_e=metrics.mean_absolute_error(trainy2_raw, train_pre_num)
print("Detect MSE:",tmse_d)
print("Estimate MSE:",tmse_e)
print("Detect RMSE:",np.sqrt(tmse_d))
print("Estimate RMSE:",np.sqrt(tmse_e))
print("Detect MAE:",tmae_d)
print("Estimate MAE:",tmae_e)

#Testing Performance
print("Detectionaccuracy_score:", accuracy_score(testy1_raw, pre))
print("Estimationaccuracy_score:", accuracy_score(testy2_raw, pre_num))
mse_d=metrics.mean_squared_error(testy1_raw, pre)
mse_e=metrics.mean_squared_error(testy2_raw, pre_num)
mae_d=metrics.mean_absolute_error(testy1_raw, pre)
mae_e=metrics.mean_absolute_error(testy2_raw, pre_num)
print("Detect MSE:",mse_d)
print("Estimate MSE:",mse_e)
print("Detect RMSE:",np.sqrt(mse_d))
print("Estimate RMSE:",np.sqrt(mse_e))
print("Detect MAE:",mae_d)
print("Estimate MAE:",mae_e)

#Consusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

CM_f1 = confusion_matrix(pre_num, testy2_raw)
# Precise matrix
sum_true = np.expand_dims(np.sum(CM_f1, axis=1), axis=1)
CM_f1 = CM_f1 / sum_true

plt.figure(figsize=(10, 8))
f, ax = plt.subplots(figsize=(10, 8))
h = sns.heatmap(CM_f1, annot=True, linewidths=2,
                annot_kws={'size': 20}, fmt='.2%', cmap="YlGnBu")
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
。

ax.tick_params(labelsize=20)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.ylabel('Estimation', fontsize=20)
plt.xlabel('Groundtruth', fontsize=20)
plt.show()