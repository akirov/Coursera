import keras
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import tensorflow as tf

print("We're using TF", tf.__version__)
print("We are using Keras", keras.__version__)

import sys

def load_dataset(flatten=False):
    (X_trn, y_trn), (X_tst, y_tst) = keras.datasets.mnist.load_data()

    # normalize x
    X_trn = X_trn.astype(float) / 255.
    X_tst = X_tst.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_trn, X_vld = X_trn[:-10000], X_trn[-10000:]
    y_trn, y_vld = y_trn[:-10000], y_trn[-10000:]

    if flatten:
        X_trn = X_trn.reshape([X_trn.shape[0], -1])
        X_vld = X_vld.reshape([X_vld.shape[0], -1])
        X_tst = X_tst.reshape([X_tst.shape[0], -1])

    return X_trn, y_trn, X_vld, y_vld, X_tst, y_tst


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# X contains rgb values divided by 255
print("X_train [shape %s] sample patch:\n" % (str(X_train.shape)), X_train[1, 15:20, 5:10])
print("A closeup of a sample patch:")
plt.imshow(X_train[1, 15:20, 5:10], cmap="Greys")
plt.show()
print("And the whole sample:")
plt.imshow(X_train[1], cmap="Greys")
plt.show()
print("y_train [shape %s] 10 samples:\n" % (str(y_train.shape)), y_train[:10])


# flatten images
X_train_flat = X_train.reshape((X_train.shape[0], -1))
print(X_train_flat.shape)

X_val_flat = X_val.reshape((X_val.shape[0], -1))
print(X_val_flat.shape)


# one-hot encode the target
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_val_oh = keras.utils.to_categorical(y_val, 10)

print(y_train_oh.shape)
print(y_train_oh[:3], y_train[:3])


# building a model with Keras
from keras.layers import Dense, Activation
from keras.models import Sequential

# we still need to clear a graph though
s = tf.InteractiveSession()

model = Sequential()  # it is a feed-forward network without loops like in RNN
model.add(Dense(256, input_shape=(784,)))  # the first layer must specify the input shape (replacing placeholders)
model.add(Activation('sigmoid'))
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))


# you can look at all layers and parameter count
model.summary()


# now we "compile" the model specifying the loss and optimizer
model.compile(
    loss='categorical_crossentropy', # this is our cross-entropy
    optimizer='adam',
    metrics=['accuracy']  # report accuracy during training
)


# and now we can fit the model with model.fit()
# and we don't have to write loops and batching manually as in TensorFlow
model.fit(
    X_train_flat, 
    y_train_oh,
    batch_size=512, 
    epochs=40,
    validation_data=(X_val_flat, y_val_oh),
#    callbacks=[keras_utils.TqdmProgressCallback()],
    verbose=1
)


# Here're the notes for those who want to play around here
# Here are some tips on what you could do:
# 
# Network size
# - More neurons,
# - More layers, (https://keras.io/)
# - Other nonlinearities in the hidden layers:
#   tanh, relu, leaky relu, etc
# - Larger networks may take more epochs to train, so don't discard your net
#   just because it could didn't beat the baseline in 5 epochs.
#
# Early Stopping
# - Training for 100 epochs regardless of anything is probably a bad idea.
# - Some networks converge over 5 epochs, others - over 500.
# - Way to go: stop when validation score is 10 iterations past maximum
#
# Faster optimization
# - rmsprop, nesterov_momentum, adam, adagrad and so on.
#   -- Converge faster and sometimes reach better optima
#   -- It might make sense to tweak learning rate/momentum, other learning
#      parameters, batch size and number of epochs
#
# Regularize to prevent overfitting
# - Add some L2 weight norm to the loss function, theano will do the rest
#   -- Can be done manually or via - https://keras.io/regularizers/
#
# Data augmemntation - getting 5x as large dataset for free is a great deal
# - https://keras.io/preprocessing/image/
# - Zoom-in+slice = move
# - Rotate+zoom(to remove black stripes)
# - any other perturbations
# - Simple way to do that (if you have PIL/Image):
#   -- from scipy.misc import imrotate,imresize
#   -- and a few slicing
# - Stay realistic. There's usually no point in flipping dogs upside down as
#   that is not the way you usually see them.
