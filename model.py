import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer

MOVING_AVERAGE_DECAY = 0.9999


def inference(images, num_classes, reuse=False):
    def _variable_with_weight_decay(name, shape, wd=0.0):
        var = tf.get_variable(name, shape=shape, initializer=xavier_initializer())
        if wd > 0.0 and not reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _activation_summary(x):
        if not reuse:
            tensor_name = x.op.name
            tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    with tf.variable_scope('conv1', reuse=reuse) as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 3, 40], initializer=xavier_initializer())
        biases = tf.get_variable('biases', shape=[40], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        _activation_summary(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2', reuse=reuse) as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 40, 60], initializer=xavier_initializer())
        biases = tf.get_variable('biases', shape=[60], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        _activation_summary(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3', reuse=reuse) as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 60, 90], initializer=xavier_initializer())
        biases = tf.get_variable('biases', shape=[90], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        _activation_summary(conv3)
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('conv4', reuse=reuse) as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 90, 90], initializer=xavier_initializer())
        biases = tf.get_variable('biases', shape=[90], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        _activation_summary(conv4)
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    with tf.variable_scope('fc5', reuse=reuse) as scope:
        reshape = tf.reshape(pool4, [images.get_shape()[0].value, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 150], wd=0.001)
        biases = tf.get_variable('biases', shape=[150], initializer=tf.constant_initializer(0.0))
        fc5 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, weights), biases), name=scope.name)
        _activation_summary(fc5)

    with tf.variable_scope('fc6', reuse=reuse) as scope:
        weights = _variable_with_weight_decay('weights', shape=[150, 150], wd=0.001)
        biases = tf.get_variable('biases', shape=[150], initializer=tf.constant_initializer(0.0))
        fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc5, weights), biases), name=scope.name)
        _activation_summary(fc6)

    with tf.variable_scope('fc7', reuse=reuse) as scope:
        weights = _variable_with_weight_decay('weights', shape=[150, num_classes])
        biases = tf.get_variable('biases', shape=[num_classes], initializer=tf.constant_initializer(0.0))
        fc7 = tf.nn.bias_add(tf.matmul(fc6, weights), biases, name=scope.name)

    return fc7


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)

    # Apply gradients, and add histograms
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer()
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op
