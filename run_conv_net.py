import  numpy  as  np
import tensorflow as tf
import get_dataset as gd

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Input
print("Starting input...")
x = tf.placeholder(tf.float32, [None, 76800])
y_ = tf.placeholder(tf.float32, [None, 2])
x_image = tf.reshape(x, [-1, 240, 320, 1])

# First conv layer
print("Conv 1...")
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second conv layer
print("Conv 2...")
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected layer
print("Densely connected layer...")
W_fc1 = weight_variable([60 * 80 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 60 * 80 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
print("Dropout")
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout 
print("Readout")
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	print("Starting session...")
	sess.run(tf.global_variables_initializer())
	# Note: Example only includes subset (25) of range
	# for live demo. For general use, use 20,000.
	for i in range(25): 
		batch_xs, batch_ys = gd.gen.next()
		train_accuracy = accuracy.eval(feed_dict={
			x: batch_xs, y_: batch_ys, keep_prob: 1.0})
		print('step %d, training accuracy %g' % (i, train_accuracy))
		train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

	print("Getting test accuracy...")
	test_imgs = gd.get_test_imgs()
	test_labels = gd.get_test_labels()
	print(test_imgs[0].shape)
	print('test accuracy %g' % accuracy.eval(feed_dict={
		x: test_imgs, y_: test_labels, keep_prob: 1.0}))


  