'''
buil a DNN model to distinguish horse and human
data source:
wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip -O /tmp/horse-or-human.zip
'''

import os
import zipfile

print("extracing horse-or-human.zip...")
local_zip  = "./tmp/horse-or-human.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall("./tmp/horse-or-human")
zip_ref.close()
print("./tmp/horse-or-human/horse and human subdir have been created")
# NOTE : we do not label jpg data as human, horse, but it will recognized from the sub-directory names by using ImageGenerator

# Directory with training horse/ human pictures
train_horse_dir = os.path.join("./tmp/horse-or-human/horses")
train_human_dir = os.path.join("./tmp/horse-or-human/humans")

# filenames look like..
train_horse_names = os.listdir(train_horse_dir)
#print(train_horse_names[:10])
train_human_names = os.listdir(train_human_dir)
#print(train_human_names[:10])

print("Total training horse images:", len(train_horse_names))
print("Total training human images:", len(train_human_names))

#%matplotlib inline # for notebook
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
# to display pictures
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
'''

# building a model
import tensorflow as tf
model = tf.keras.models.Sequential([
    # input shape : image 300 x 300 with 3 bytes color
    # first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    # second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layers
    tf.keras.layers.Dense(512, activation='relu'),
    # only 1 output neron. contains a value from 0-1 (0=horse, 1=human)
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

# data preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# all images will be rescaled by 1/255 (normalizaing)
train_datagen = ImageDataGenerator(rescale=1/255)

# flow training images in batches of 128 using train_datagen generator
''' NOTE
This ImageDataGenerator class allows you to instantiate generators of augmented image batches (and their labels) via
(1).flow(data, labels) or
(2).flow_from_directory(directory).
'''
train_generator = train_datagen.flow_from_directory(
    '/Users/gina/Desktop/projects/DeepLearning/VisionNN/tmp/horse-or-human', #source of training images
    target_size=(300, 300),
    batch_size=128,
    #since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# training
history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=3,
    verbose=1) # verbose = 0 for silent, verbose = 1 for progressing bar

# running a prediction using the model
print("Running a prediction using the model...")

import numpy as np
from tensorflow.keras.preprocessing import image


# predicting images
path = "/Users/gina/Desktop/projects/DeepLearning/VisionNN/content/"

test_files = os.listdir(path)
print("test files:", test_files)

for fn in test_files:
    img = image.load_img(path + fn, target_size=(300,300))
    x = image.img_to_array(img) #shape: (300,300,3)
    x = np.expand_dims(x, axis=0) #shape: (1,300,300,3)

    images = np.vstack([x]) #shape: (300,300,3)
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(fn + " is a human")
    else:
        print(fn + " is a horse")

# visualizing intermediate representations
# NOTE: to get a feel for what kind of features our convnet has learnt,
# visualize how an input gets transformed as it goes through the convnet.
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# define a new model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after the first

successive_outputs = [layer.output for layer in model.layers[1:]] #model have 11 layers

visualization_model = tf.keras.models.Model(inputs = model.input,
                                            outputs = successive_outputs)

# prepare a random input image from the training set
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files) # choose 1 randomly

img = load_img(img_path, target_size=(300,300))
x = img_to_array(img) # numpy array with shape (300, 300, 3)
x = x.reshape((1,) + x.shape) # numpy array with shape (1,300,300,3)

# rescale by 1/255
x /= 255

# obtaining all intermediate representations for this image
successive_feature_maps = visualization_model.predict(x)
print("successive_feature_maps list length:", len(successive_feature_maps))

# layers name
layer_names = [layer.name for layer in model.layers]
print("layers name:", layer_names) #['conv2d', 'max_pooling2d', 'conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2', 'conv2d_3', 'max_pooling2d_3', 'flatten', 'dense', 'dense_1']

# display
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        # just do this for the conv/max pool layers, not fully-connected layers
        n_features = feature_map.shape[-1] # nums of features in feature map
        # the feature map has shape( 1, size, size, n_features)
        size = feature_map.shape[1]
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            # Postprocess the feature to make it visually palatable
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            # We'll tile each filter into this big horizontal grid
            display_grid[:, i * size : (i + 1) * size] = x
        # Display the grid
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
# NOTE:As you can see we go from the raw pixels of the images to increasingly abstract and compact representations. The representations downstream start highlighting what the network pays attention to, and they show fewer and fewer features being "activated"; most are set to zero. This is called "sparsity." Representation sparsity is a key feature of deep learning.
print("These representations carry increasingly less information about the original pixels of the image, but increasingly refined information about the class of the image. You can think of a convnet (or a deep network in general) as an information distillation pipeline. ")

# clean up to terminate the kernel and free memory resources:
input_ = input("Termintate the kernel and free memory resource? (y/n):")
if input_ == "y":
    import os, signal
    os.kill(os.getpid(), signal.SIGKILL)
