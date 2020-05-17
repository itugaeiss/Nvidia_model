from keras.models import load_model
import numpy as np
from keras.optimizers import Adam

x = np.load("X_train.data.npy")
y = np.load("y_train.data.npy")


model = load_model("selfv1.model")
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00001),metrics=["accuracy"])
model.fit(x,y,batch_size=40,epochs=10,validation_split=0.2)
model.save("selfv1.2.model")