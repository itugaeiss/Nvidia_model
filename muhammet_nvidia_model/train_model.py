from tensorflow.keras import models
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import utils 

def load_data():#tek ve üçlü kameraya göre tekrar tekrar at
	adress = os.path.join("C:/python/keras/self_driving/beta_simulator_windows/data","driving_log.csv")
	print(adress)
	df = pd.read_csv(adress)

	X = df['center'].values
	#X = df["center"].values
	y = df["steering"].values
	print(y)
	X_train_name, X_test_name, y_train, y_test = train_test_split(X,y)
	
	X_train = []
	X_test = []
	

	for center in X_train_name:
		image_c = cv2.imread(center)
		X_train.append(utils.preprocess(image_c))
	
	for center in X_test_name:
		image_c = cv2.imread(center)
		X_test.append(utils.preprocess(image_c))

	X_train = np.array(X_train)
	X_test = np.array(X_test)

	np.save("X_train.data",X_train)
	np.save("X_test.data",X_test)
	np.save("y_train.data",y_train)
	np.save("y_test.data",y_test)

	return X_train,X_test,y_train,y_test

def build_model():
	model=Sequential()
	model.add(Lambda(lambda x: x/127.5-1.0,input_shape=(160,320,3)))
	#ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
	model.add(Conv2D(24,5,5,activation="elu",subsample=(2,2)))
	model.add(Conv2D(36,5,5,activation="elu",subsample=(2,2)))
	model.add(Conv2D(48,5,5,activation="elu",subsample=(2,2)))
	model.add(Conv2D(64,5,5,activation="elu"))
	model.add(Conv2D(64,5,5,activation="elu"))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100,activation="elu"))
	model.add(Dense(50,activation="elu"))
	model.add(Dense(10,activation="elu"))
	model.add(Dense(1))

	model.summary()
	return model

"""X_train = np.load("X_train.data.npy")
y_train = np.load("y_train.data.npy")"""
X_train,_,y_train, _ = load_data()
print(y_train)
"""
print("X_train ",X_train.shape,"y_train ",y_train.shape)
model = build_model()
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001),metrics=["accuracy"])
model.fit(X_train,y_train,batch_size=40,epochs=15,validation_split=0.1)
model.save("selfv1.model")"""