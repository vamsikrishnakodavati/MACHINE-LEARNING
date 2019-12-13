from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
savePath = 'tmp/tensor_model'
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10]) 
Weight = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
y = tf.nn.softmax(tf.matmul(x,Weight)+bias)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(10000):
    batch = mnist.train.next_batch(100)  
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
acc_eval = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Current accuracy: %.2f%%"% (acc_eval*100))
saver.save(sess, savePath)
print('Session saved in path '+savePath)
