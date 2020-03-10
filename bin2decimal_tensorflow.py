import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np


def bin2int(bin_list):
	bin_str=""
	for k in bin_list:
		bin_str += str(int(k))

	return int(bin_str,2)


def dataset(num,bin_len):
	x = np.zeros((num,bin_len)) #x = binary number,y = label
	y = np.zeros((num))

	for i in range(num):
		x[i] = np.round(np.random.rand(bin_len)).astype(int)
		y[i] = bin2int(x[i])
		#print(x[i],":",y[i])

	return x,y

import tensorflow as tf
from tensorflow.contrib import rnn

x_len = 8
no_of_samples = 1000
lr = 0.001
training_steps = 22000
display_step = 1000

n_input = x_len
timestep = 1
n_hidden = 16
n_output = 1
test_sample = 10

trainX,trainY = dataset(no_of_samples,x_len)
testX,testY = dataset(test_sample,x_len)

#graph_input
X = tf.placeholder(tf.float32,[None,timestep,n_input])
Y = tf.placeholder(tf.float32,[None,n_output])

weights = tf.Variable(tf.random_normal([n_hidden,n_output]))
bias = tf.Variable(tf.random_normal([n_output]))

def RNN(x,W,b):

	x = tf.unstack(x,timestep,1)	#decreases dimension of tensor x from R to R-1
	lstm_cell = rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
	outputs,states = rnn.static_rnn(lstm_cell,x,dtype = tf.float32)

	return tf.matmul(outputs[-1],W)+b

logits = RNN(X,weights,bias)

trainX = np.reshape(trainX,[-1,timestep,n_input])
trainY = np.reshape(trainY,[-1,n_output])


testX = np.reshape(testX,[-1,timestep,n_input])
testY = np.reshape(testY,[-1,n_output])

loss = tf.reduce_mean(tf.losses.mean_squared_error(logits,Y))
optimizer = tf.train.RMSPropOptimizer(lr)
train = optimizer.minimize(loss)

with tf.Session() as sess:
	tf.global_variables_initializer().run()

	for step in range(training_steps):
		_,_loss = sess.run([train,loss],feed_dict = {X:trainX,Y:trainY})

		if (step+1)%display_step == 0:
			print("Step: ",step+1,"\tLoss: ",_loss)

	print("Optimization Finished!!!")


	result = sess.run(logits, feed_dict={X: testX})
	result = sess.run(tf.round(result))

	print("Real \t\t\t\t Guess")
	for i in range(test_sample):
		if testY[i] == result[i]:
			print("True")
		else:
			print("False")	
		print(testY[i], ' -> ', result[i])











