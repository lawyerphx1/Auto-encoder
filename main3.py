import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

IMAGE_SIZE_MNIST = 28

def encoder(weights, biases, x):
    L1 = tf.nn.sigmoid( tf.matmul(x, weights['encoder_h1']) + biases['encoder_h1'] )
    L2 = tf.nn.sigmoid( tf.matmul(L1,weights['encoder_h2']) + biases['encoder_h2'] )
    return L2

def decoder(weights, biases, x):
    L1 = tf.nn.sigmoid( tf.matmul(x, weights['decoder_h1']) + biases['decoder_h1'] )
    L2 = tf.nn.sigmoid( tf.matmul(L1,weights['decoder_h2']) + biases['decoder_h2'] )
    return L2

def main():
    examples_to_show = 10
    learning_rate = 0.01
    batch_size = 256
    display_step = 1
    n_input = IMAGE_SIZE_MNIST ** 2
    n_hidden_1 = 256
    n_hidden_2 = 128
    train_eco = 7
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_input]))
    }
    biases = {
        'encoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_input]))
    }
    X = tf.placeholder(tf.float32, [None,n_input])
    Y = decoder(weights,biases,encoder(weights,biases,X))
    Y_true = X
    loss = tf.reduce_mean(tf.square(Y_true - Y))
    solve = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = int(mnist.train.num_examples / batch_size)
        for epoch in range(train_eco):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _,c = sess.run([solve,loss], feed_dict={X:batch_xs})
            if epoch % display_step == 0:
                print("epoch:",epoch+1, "cost = ", c)
        #show the map?
        encode_decode = sess.run(Y, feed_dict={X: mnist.test.images[:examples_to_show]})
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)))
            a[1][i].imshow(np.reshape(encode_decode[i], (IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)))
        plt.show()

if __name__ == '__main__':
    main()