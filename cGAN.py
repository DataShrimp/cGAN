import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

# Hyperparameter
x_dim = mnist.train.images.shape[1]
y_dim = 1  # 0~9 digits
z_dim = 100
h_dim = 128
alpha = 0.01
smooth = 0.1
lr = 0.002
batch_size = 100
epochs = 100

x = tf.placeholder(tf.float32, [None, x_dim])
y = tf.placeholder(tf.float32, [None, y_dim])
z = tf.placeholder(tf.float32, [None, z_dim])

def generator(z, y, out_dim, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope("generator", reuse=reuse):
        inputs = tf.concat(axis=1, values=[z, y])
        h1 = tf.layers.dense(inputs, n_units, activation=None)
        h1 = tf.maximum(alpha*h1, h1)

        logits = tf.layers.dense(h1, out_dim)
        out = tf.tanh(logits)
        return out

def discriminator(x, y, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope("discriminator", reuse=reuse):
        inputs = tf.concat(axis=1, values=[x, y])
        h1 = tf.layers.dense(inputs, n_units, activation=None)
        h1 = tf.maximum(alpha*h1, h1)

        logits = tf.layers.dense(h1, 1)
        out = tf.tanh(logits)
        return logits, out

# Build net
g_model = generator(z, y, x_dim)
d_model_real, d_logits_real = discriminator(x, y)
d_model_fake, d_logits_fake = discriminator(g_model, y, reuse=True)

# Calculate losses
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real) * (1 - smooth)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_real)))

d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

# Get the trainable_var
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(lr).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(lr).minimize(g_loss, var_list=g_vars)

# Training
samples = []
losses = []
# Save generator variables
saver = tf.train.Saver(var_list=g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            batch_x = batch[0].reshape((batch_size, x_dim))
            batch_x = batch_x * 2 - 1
            batch_y = batch[1].reshape((batch_size, y_dim))
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
            sess.run(d_train_opt, feed_dict={x: batch_x, y: batch_y, z:batch_z})
            sess.run(g_train_opt, feed_dict={z: batch_z, y: batch_y})

        # End of the loss
        train_loss_d = sess.run(d_loss, feed_dict={x: batch_x, y: batch_y, z:batch_z})
        train_loss_g = sess.run(g_loss, feed_dict={z: batch_z, y: batch_y})

        print("Epoch {}/{}...".format(e+1, epochs), "d-loss {:.4f}".format(train_loss_d), "g-loss {:.4f}".format(train_loss_g))
        losses.append((train_loss_d, train_loss_g))
        saver.save(sess, "./checkpoints/generator.ckpt")

if __name__ == "__main__":
    pass
