# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:25:50 2017

@author: Quentin
"""

import tensorflow as tf
import numpy as np

#from tensorflow.contrib.layers import fully_connected



#CONSTRUCTION PHASE

n_inputs = 28*28 #MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape = (None), name="y")

#create 1 layer
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name): #create a namescope using the name of the layer
        n_inputs = int(X.get_shape()[1]) #get the number of inputs by looking up into the matrix shape
        stddev = 2 / np.sqrt(n_inputs) 
        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
        W = tf.Variable(init, name="weights") 
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        z = tf.matmul(X, W) + b #create a subgraph to compute z
        if activation == "relu" :
            return tf.nn.relu(z)
        else:
            return z
        
        
        
#create deep nn
with tf.name_scope("dnn"):    
    hidden1 = neuron_layer(X, n_hidden1, "nhidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")
    
#with tf.name_scope("dnn"):
#    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
#    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
#    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)
    
    
#define the cost function that will be used to train nn
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    
#define the optimizer
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
#specify the evaluatation function
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
# create node to initialize all variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#EXECUTION PHASE

#load mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/data/")
    
#define the number of epochs we want to run, as well as the size of the mini-batches
n_epochs= 20
batch_size = 50


#train the model
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:X_batch, y : y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X : mnist.test.images, y : mnist.test.labels})
        print(epoch, "Train accuracy : ", acc_train, " Test accuracy: ", acc_test)
    save_path = saver.save(sess, "./my_model_final.ckpt")    
    
#make predictions    
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt") # or better, use save_path
    X_new_scaled = mnist.test.images[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
    
print("Predicted classes:", y_pred)
print("Actual classes:   ", mnist.test.labels[:20])

