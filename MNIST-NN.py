from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batchSize = 2250
batch = mnist.train.next_batch(batchSize)

x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None, 10])

"""Activations in the 1st layer after applying the sigmoid function.
The hidden layer contains 30 neurons. """

w1 = tf.Variable(tf.random_normal([784,30])/batchSize**0.5)
b1 = tf.Variable(tf.random_normal([1]))

a1 = tf.nn.sigmoid(tf.matmul(x,w1) + b1)

""" This computes the output of the network using the softmax classifier """
w2 = tf.Variable(tf.random_normal([30,10])/batchSize**0.5)
b2 = tf.Variable(tf.random_normal([1]))

yt = tf.nn.softmax(tf.matmul(a1,w2) + b2)

# using the l2 norm (without regularization)
lossBatch = tf.nn.l2_loss(y-yt)/(2*batchSize)

#regularization term with the r. parameter = 0.81
reg = 0.81*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))/batchSize

#loss with the regularization term
loss = lossBatch + reg

train = tf.train.GradientDescentOptimizer(6.8).minimize(loss)

accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(yt,1),tf.argmax(y,1)),tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range (2200):        
        sess.run(train, {x:batch[0], y:batch[1]})

##    print (sess.run(lossBatch, {x:batch[0], y:batch[1]}))    
    print(sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}))

##st = ""
##for i in range(784):
##    if i%28 == 0:
##        st += '\n'
##    st = st + str(np.around(batch[0][0][i],1)) + "  "
##
##print (st)
