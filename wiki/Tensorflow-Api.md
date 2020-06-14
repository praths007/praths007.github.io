## Table of Contents
* [Sample Code](#sample-code)
* [TensorFlow](#tensorflow)
   * [General](#general-1)
   * [Data Preprocessing](#data-preprocessing)
   * [Model Selection](#model-selection)
   * [Accuracy & Predictions](#accuracy--predictions)
   * [Models](#models)



## Sample Code

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_hl1, n_nodes_hl2, n_nodes_hl3 = 500, 500, 500
n_classes = 10
batch_size = 100

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")


def neural_network_model(data):

    # height * width
    hidden_1_layer = {"weights": tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    "biases": tf.Variable(tf.random_normal([n_classes]))}

    # model = (input_data * weight) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]), hidden_1_layer["biases"])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer["weights"]), hidden_2_layer["biases"])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer["weights"]), hidden_3_layer["biases"])
    l3 = tf.nn.relu(l3)

    op = tf.matmul(l3, output_layer["weights"]) + output_layer["biases"]
    # op = tf.nn.sigmoid(op)
    return op



def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x_epoch, y_epoch = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: x_epoch, y: y_epoch})
                epoch_loss += c
            print('Epoch', epoch, "completed out of", epochs, "loss:", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)
```