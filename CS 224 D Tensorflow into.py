# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

a=tf.constant(5)
b=tf.constant(5)
c=a*b

with tf.Session() as sess:
	print(sess.run(c))
	x=c.eval()
	print(c.eval())


print(x)

W1 = tf.ones((2,2))
W2 = tf.Variable(tf.zeros((2,2)), name="weights")
with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())
    print(sess.run(W2))
    print(sess.run(W1))




#### Updating variable
state = tf.Variable(0, name="counter")
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)


with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(state))
	for _ in range(3):
         sess.run(update)
         print(sess.run(state))


###Fetching Variable State (1)
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
	result = sess.run([mul, intermed])
	print("result is      :"+str(result))


###Inputting Data
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
	print(sess.run(ta))

### Placeholders and Feed Dictionaries (2)
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

#### Placeholders and Feed Dictionaries (2)
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)
with tf.Session() as sess:
	print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))



with tf.variable_scope("foo"):
 with tf.variable_scope("bar"):
    v = tf.get_variable("v", [1])

#            assert v.name == "foo/bar/v:0



with tf.variable_scope("foo"):
 v = tf.get_variable("v", [1])
 tf.get_variable_scope().reuse_variables()
 v1 = tf.get_variable("v", [1])
assert v1 == v



### Ex: Linear Regression in TensorFlow (1)
import numpy as np
import seaborn
import matplotlib.pyplot as plt
# Define input data
X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)
# Plot input data
#plt.scatter(X_data, y_data)


# Define data size and batch size
n_samples = 1000
batch_size = 1000
# Tensorflow is finicky about shapes, so resize
X_data = np.reshape(X_data, (n_samples,1))
y_data = np.reshape(y_data, (n_samples,1))
# Define placeholders for input
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))



# Define variables to be learned
with tf.variable_scope("linear-regression"):
 W = tf.get_variable("weights", (1, 1),
 initializer=tf.random_normal_initializer())
 b = tf.get_variable("bias", (1,),
 initializer=tf.constant_initializer(0.0))
 y_pred = tf.matmul(X, W) + b
 loss = tf.reduce_sum((y - y_pred)**2/n_samples)

opt = tf.train.AdamOptimizer()
opt_operation = opt.minimize(loss)
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	sess.run([opt_operation], feed_dict={X: X_data, y: y_data})

# Sample code to run full gradient descent:
# Define optimizer operation
opt_operation = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
 # Initialize Variables in graph
	sess.run(tf.initialize_all_variables())
 # Gradient descent loop for 500 steps
	for _ in range(500):
 # Select random minibatch
		indices = np.random.choice(n_samples, batch_size)
		X_batch, y_batch = X_data[indices], y_data[indices]
 # Do gradient descent step
		_, loss_val = sess.run([opt_operation, loss], feed_dict={X: X_batch, y: y_batch})