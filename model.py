import tensorflow as tf

MOVING_AVERAGE_DECAY = 0.9999


def inference(images, num_classes, reuse=False):
    def _activation_summary(x):
        if not reuse:
            tensor_name = x.op.name
            tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    inputs = tf.identity(images, name='inputs')

    with tf.variable_scope('conv1', reuse=reuse) as scope:
        conv1 = tf.layers.conv2d(inputs, 60, [3, 3], padding='SAME', activation=tf.nn.relu)
        _activation_summary(conv1)
        conv1 = tf.identity(conv1, name=scope.name)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2', reuse=reuse) as scope:
        conv2 = tf.layers.conv2d(pool1, 90, [3, 3], padding='SAME', activation=tf.nn.relu)
        _activation_summary(conv2)
        conv2 = tf.identity(conv2, name=scope.name)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3', reuse=reuse) as scope:
        conv3 = tf.layers.conv2d(pool2, 120, [3, 3], padding='SAME', activation=tf.nn.relu)
        _activation_summary(conv3)
        conv3 = tf.identity(conv3, name=scope.name)
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('conv4', reuse=reuse) as scope:
        conv4 = tf.layers.conv2d(pool3, 150, [3, 3], padding='SAME', activation=tf.nn.relu)
        _activation_summary(conv4)
        conv4 = tf.identity(conv4, name=scope.name)
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    with tf.variable_scope('fc5', reuse=reuse) as scope:
        reshape = tf.reshape(pool4, [images.get_shape()[0].value, -1])
        fc5 = tf.layers.dense(reshape, 200, activation=tf.nn.relu)
        _activation_summary(fc5)
        fc5 = tf.identity(fc5, name=scope.name)

    with tf.variable_scope('fc6', reuse=reuse) as scope:
        fc6 = tf.layers.dense(fc5, 200, activation=tf.nn.relu)
        _activation_summary(fc6)
        fc6 = tf.identity(fc6, name=scope.name)

    with tf.variable_scope('fc7', reuse=reuse) as scope:
        fc7 = tf.layers.dense(fc6, num_classes, activation=None)
        fc7 = tf.identity(fc7, name=scope.name)

    return fc7


def loss(logits, labels):
    # cross entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', mean)
    # add weight decay
    wd = {
        'fc5': 0.001,
        'fc6': 0.001,
    }
    for scope, scale in wd.items():
        with tf.variable_scope(scope, reuse=True):
            v = tf.get_variable('dense/kernel')
            weight_decay = tf.multiply(tf.nn.l2_loss(v), scale, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
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
