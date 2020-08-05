# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import os
import math


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/", one_hot=True, reshape=False)
# The one_hot=True argument specifies that instead of the labels associated with each
# image being the digit itself i.e. "4", it is a vector with "one hot" node and all
# the other nodes being zero i.e. [0, 0, 0, 0, 1, 0, 0, 0, 0, 0].

# Or, to suppress TF warning do:
#mnist = tf.keras.datasets.mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Declare the training data placeholders
# input x - for 28 x 28 grayscale pixels
X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')

# Correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])

# step for variable learning rate
step = tf.placeholder(tf.int32, name='step')

# For the dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 10 softmax neurons)
K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer


# The model
W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)

W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)

W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])
Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)

# Add dropout? See https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow
layer_drop = tf.nn.dropout(Y4, keep_prob)

# And the weights connecting the last hidden layer to the output layer
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)
# Output layer (predictions) - use a softmax activation
#Ylogits = tf.add(tf.matmul(layer_drop, W5), B5, name='output')
Ylogits = tf.add(tf.matmul(Y4, W5), B5, name='output')
Y = tf.nn.softmax(Ylogits)


# Here we'll use the cross entropy loss function, represented by:
# J = −1/m ∑i=1..m ∑j=1..n y_j^(i) log(y_j_^(i)) + (1–y_j^(i)) log(1–y_j_^(i))
# Where y_j^(i) is the ith training label for output node j,
# y_j_^(i) is the ith predicted label for output node j, m is the number of training / batch samples
# and n is the number of digits.
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100


# accuracy of the trained model, between 0 (worst) and 1 (best)
# The tf.argmax function is the same as the numpy argmax function, which returns
# the index of the maximum value in a vector / tensor.  Therefore, the
# correct_prediction operation returns a tensor of size (m x 1) of True and False
# values designating whether the neural network has correctly predicted the digit.
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Optimisation variables
epochs = 100
batch_size = 100
dropout = 0.75


# Add an optimiser. Can be tf.train.GradientDescentOptimizer.
# In very high dimensional spaces like here - we have in the order of 10K weights and biases -
# "saddle points" are frequent. These are points that are not local minima but where the gradient
# is nevertheless zero and the gradient descent optimizer stays stuck there. Adam is better.
# Training step, the learning rate is a placeholder
# the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
stepmod = tf.placeholder(tf.float32, name='stepmod')  # will be 0 or 1
lr = 0.0001 +  stepmod * tf.train.exponential_decay(0.005, step, 1, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


# Finally setup the initialisation operator
init_op = tf.global_variables_initializer()


# Create a saver object which will save all the variables
saver = tf.train.Saver(max_to_keep=5)


from matplotlib import pyplot as plt


from timeit import default_timer as timer
tstart = timer()

# Start the training session
with tf.Session() as sess:
    imported_graph = None
    if os.path.isfile(os.path.abspath("cnn_model.meta")):
        imported_graph = tf.train.import_meta_graph(os.path.abspath('cnn_model.meta'))  # Don't need this probably

    if imported_graph is not None:
        print("Loading saved model..")
        imported_graph.restore(sess, os.path.abspath("cnn_model"))  # Or use the saver
        smod = 0.0
    else:
        print("Initializing variables...")
        sess.run(init_op)  # Initialise the variables
        smod = 1.0

    test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, keep_prob: 1.0})
    print("Before training! Test accuracy = ", test_acc)

    if test_acc < 0.99 :  # Only train if not trained already
        total_batch = int(len(mnist.train.labels) / batch_size)
        losses = []
        for epoch in range(epochs):  # Or require some accuracy to stop?
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                _, ce = sess.run([train_step, cross_entropy], feed_dict={X: batch_x, Y_: batch_y,
                                                                         stepmod : smod,
                                                                         step: epoch,
                                                                         keep_prob: dropout})
                avg_cost += ce
            avg_cost /= total_batch
            l = sess.run([lr], feed_dict={stepmod : smod, step: epoch})
            print("Training epoch:", (epoch + 1), ", loss =", "{:.5f}".format(avg_cost),
                  ", lr =", str(l[0]))
            losses.append(avg_cost)
            # Save the model every 10 epochs
            if epoch % 10 == 0 : saver.save(sess, os.path.abspath("cnn_model"))
            if avg_cost < 0.0001: break

        test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, keep_prob: 1.0})
        print("Training finished! Test accuracy = ", test_acc)

        tend = timer()
        print("Training time :", (tend - tstart))

        if len(losses) > 1:
            plt.plot(losses)
            plt.xlabel("Epochs")
            plt.ylabel("Training loss")
            plt.show()

        # Save the final model
        saver.save(sess, os.path.abspath("cnn_model"))


import cv2
import numpy as np


drawing = False # true if mouse is pressed
oldx , oldy = None , None

# mouse callback function
def line_drawing(event, curx, cury, flags, param):
    global oldx, oldy, drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        oldx, oldy = curx, cury
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img, (oldx, oldy), (curx, cury), color=255, thickness=30)
            oldx, oldy = curx, cury
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img, (oldx, oldy), (curx, cury), color=255, thickness=30)


# Reset TF graph and vars
tf.reset_default_graph()

# Import saved graph. imported_graph is a Saver
imported_graph = tf.train.import_meta_graph(os.path.abspath('cnn_model.meta'))

# Run recognition session
with tf.Session() as sess:
    # Restore saved variables
    imported_graph.restore(sess, os.path.abspath("cnn_model"))
    output_tensor = sess.graph.get_tensor_by_name('output:0')

    while True:
        # Draw with the mouse
        img = np.full((384, 384), 0, np.uint8)  # or (512, 512, 1) ?

        cv2.namedWindow('Draw a digit and press Enter to recognize. Esc to close')
        cv2.setMouseCallback('Draw a digit and press Enter to recognize. Esc to close', line_drawing)
        key = None
        while True:
            cv2.imshow('Draw a digit and press Enter to recognize. Esc to close', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == 13:
                break
        cv2.destroyAllWindows()

        if key == 27:
            break
        else:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            newimg = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            cv2.imwrite("digit.png", newimg)
            newimg = newimg.astype(float) / 255.0
            #plt.imshow(newimg, cmap="Greys")
            #plt.show()
            # Make a prediction
            newimg = newimg.reshape(28,28,1)
            predictions = sess.run(tf.nn.softmax(output_tensor), {'input:0': [newimg], 'keep_prob:0': 1.0})
            print(predictions, np.argmax(predictions))
