from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batchSize = 50
learning = 0.001

x = tf.placeholder(tf.float32, [None,784])
x_ = tf.reshape(x, [-1,28,28,1])
y = tf.placeholder(tf.float32, [None,10])

weight1 = tf.Variable(tf.random_normal([5,5,1,32]))
bias1 = tf.Variable(tf.random_normal([32]))
conv1 = tf.nn.conv2d(x_, weight1, strides = [1,1,1,1], padding = "SAME") + bias1
outputconv1 = tf.nn.relu(conv1)
maxpool1 = tf.nn.max_pool(outputconv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

weight2 = tf.Variable(tf.random_normal([5,5,32,64]))
bias2 = tf.Variable(tf.random_normal([64]))
conv2 = tf.nn.conv2d(maxpool1, weight2, strides = [1,1,1,1], padding = "SAME") + bias2
outputconv2 = tf.nn.relu(conv2)
maxpool2 = tf.nn.max_pool(outputconv2,ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

flatten = tf.reshape(maxpool2, [-1,7*7*64])
weightFor1st = tf.Variable(tf.random_normal([7*7*64,1024]))
b1 = tf.Variable(tf.random_normal([1024]))
output2ndLayer = tf.nn.relu(tf.matmul(flatten,weightFor1st) + b1)

weightfor2nd = tf.Variable(tf.random_normal([1024,10]))
b2 = tf.Variable(tf.random_normal([10]))
finaloutput = tf.nn.softmax(tf.matmul(output2ndLayer,weightfor2nd) + b2)
ffoutput = tf.argmax(finaloutput,1)

##loss = tf.nn.l2_loss(y - finaloutput)/(2*batchSize)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=finaloutput))
train = tf.train.GradientDescentOptimizer(learning).minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),ffoutput), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range (50):
        batch = mnist.train.next_batch(batchSize)
        sess.run(train, {x:batch[0], y:batch[1]})
    #prints the accuracy
    print(sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}))
