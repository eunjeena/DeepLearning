'''Define and Compile the Neural Network'''
import tensorflow as tf
import numpy as np
from tensorflow import keras

# define NN
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# compile
model.compile(optimizer='sgd', loss='mean_squared_error')

# provide data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# train
model.fit(xs, ys, epochs=500)

# predict
print(model.predict([10.0]))
# y= 2x - 1, expetcing 19
