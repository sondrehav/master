from keras.models import Model
from keras.layers import Conv1D, BatchNormalization, Activation, Flatten, Input
import numpy as np

inp =  Input(shape=(64, 1))
conv = Conv1D(filters=1, kernel_size=16, padding='same')(inp)
norm = BatchNormalization()(conv)
act = Activation('relu')(norm)
flat = Flatten()(act)
model = Model(inp, flat)
model.compile(loss='mse', optimizer='adam')
model.summary()

# get some data
X = np.expand_dims(np.random.randn(10, 64), axis=2)
Y = np.expand_dims(np.random.randn(10, 64), axis=2)

# fit model
model.fit(X, Y)