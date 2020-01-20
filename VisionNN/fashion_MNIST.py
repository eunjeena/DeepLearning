'''build graphics recognizable NN model
by using Fashion MNIST data from tf.keras datasets API'''

import tensorflow as tf
import matplotlib.pyplot as plt

# load data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

print("training images:", training_images.shape)
print("training labels:", training_labels.shape)
print("    test images:", test_images.shape)
print("    test labels:", test_labels.shape)

# see examples of data
#print("First training images")
#print(training_images[0])
#print("First training label")
#print(training_labels[0])
#print("First training image")
#plt.imshow(training_images[0])
#plt.show()

# training data has values between 0 - 255
# NOTE: it is better learning with normalizing data
training_images = training_images / 255.0
test_images = test_images / 255.0


# model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    # hidden layer
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    # output layer (10 classes)
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = tf.optimizers.Adam(), #'adam' is good too
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
# callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        '''this def is reserved'''
        # logs looks like {'loss': 0.5006339710474015, 'accuracy': 0.8244333}
        if (logs.get("accuracy") > 0.6):
            print('\n >> Reached 60% accuracy after last epoch, so cancel training')
            self.model.stop_training = True
callbacks = myCallback()

model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

# evaluate with test data
print("\nEvaluating based on test data")
eval_ = model.evaluate(test_images, test_labels) #returns [loss, accuracy]
print("loss    :", eval_[0])
print("accuracy:", eval_[1])

# test data
predicted_class = model.predict(test_images) # (10000, 28, 28)
print('predicted class:', predicted_class.shape) # (10000, 10)
print("first predicted class is list of 10 probs")
print(predicted_class[0])


print("Using Convolution and Pooling.. Building model again..")
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# instead of 60000 28x28x1, we need 6000x28x28x1 (single 4D list)
training_images=training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

training_images=training_images / 255.0
test_images=test_images/255.0

model2 = tf.keras.models.Sequential([
    # convolution and pooling
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    # 64: # of convolutions to generate. purely arbitrary, but start with 32
    # first layer should have input_shape (2x2)=4 -> 1
  tf.keras.layers.MaxPooling2D(2, 2),
    # (2 x 2) is quarter the size
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model2.summary()
model2.fit(training_images, training_labels, epochs=5)
eval2_ = model2.evaluate(test_images, test_labels)
# compare
print("compare between models with convolution/pooling and without..")
print("loss:", eval_[0], eval2_[0])
print("accracy:", eval_[1], eval2_[1])

takeaways = '''
=====================================================================
1) Adding more neurons (than 128), more calculation, slow learning,
but more accurate based on training data.

2) Input layer should have same shape of the data. 28*28 input would be infeasible, so better to 'flatten' into 28*28=784*1 by using Flatten().

3) Increasing epochs has decreasing loss, but it could overfit the data.
=====================================================================
'''
print(takeaways)
