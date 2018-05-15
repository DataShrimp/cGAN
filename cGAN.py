import pickle as pkl
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')


def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real')
    inputs_y = tf.placeholder(tf.float32, (None, 1), name="input_y")
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

    return inputs_real, inputs_y, inputs_z


def generator(z, y, out_dim, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        # combine inputs
        inputs = tf.concat(axis=1, values=[z, y])
        # Hidden layer
        h1 = tf.layers.dense(inputs, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)
        # Logits and tanh output
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.tanh(logits)

        return out


def discriminator(x, y, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        # combine inputs
        inputs = tf.concat(axis=1, values=[x,y])
        # Hidden layer
        h1 = tf.layers.dense(inputs, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)
        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)

        return out, logits

# Size of input image to discriminator
input_size = 784
# Size of latent vector to generator
z_size = 100
# Size of label for both g/d
y_size = 1
# Sizes of hidden layers in generator and discriminator
g_hidden_size = 128
d_hidden_size = 128
# Leak factor for leaky ReLU
alpha = 0.01
# Smoothing
smooth = 0.1

tf.reset_default_graph()
# Create our input placeholders
input_real, input_y, input_z = model_inputs(input_size, z_size)

# Build the model
g_model = generator(input_z, input_y, input_size, n_units=g_hidden_size, alpha=alpha)
# g_model is the generator output

d_model_real, d_logits_real = discriminator(input_real, input_y, n_units=d_hidden_size, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(g_model, input_y, reuse=True, n_units=d_hidden_size, alpha=alpha)

# Calculate losses
d_loss_real = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                          labels=tf.ones_like(d_logits_real) * (1 - smooth)))
d_loss_fake = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                          labels=tf.zeros_like(d_logits_real)))
d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                     labels=tf.ones_like(d_logits_fake)))

# Optimizers
learning_rate = 0.001

# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

batch_size = 100
epochs = 100
samples = []
losses = []

# Only save generator variables
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if 1:
        saver.restore(sess, tf.train.latest_checkpoint("checkpoints"))
    for e in range(epochs):
        for ii in range(mnist.train.num_examples // batch_size):
            batch = mnist.train.next_batch(batch_size)

            # Get images, reshape and rescale to pass to D
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images * 2 - 1

            # Get Labels
            batch_y = batch[1].reshape((batch_size, y_size))

            # Sample random noise for G
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z, input_y: batch_y})
            _ = sess.run(g_train_opt, feed_dict={input_z: batch_z, input_y: batch_y})

        # At the end of each epoch, get the losses and print them out
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images, input_y:batch_y})
        train_loss_g = g_loss.eval({input_z: batch_z, input_y:batch_y})

        print("Epoch {}/{}...".format(e + 1, epochs),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))
        # Save losses to view after training
        losses.append((train_loss_d, train_loss_g))

        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(10, z_size))
        sample_y = np.arange(10).reshape((10,1))
        gen_samples = sess.run(
            generator(input_z, input_y, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha),
            feed_dict={input_z: sample_z, input_y: sample_y})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

if __name__ == "__main__":
    pass