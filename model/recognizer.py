import tensorflow as tf
import os

class Recognizer:
    IMAGE_SIZE = 112
    INPUT_SIZE = 96
    MOVING_AVERAGE_DECAY = 0.9999

    def __init__(self, batch_size=128):
        self.batch_size = batch_size

    def inputs(self, files, num_examples_per_epoch_for_train=5000):
        queues = {}
        for i in range(len(files)):
            key = i % 5
            if key not in queues:
                queues[key] = []
            queues[key].append(files[i])

        def read_files(files):
            fqueue = tf.train.string_input_producer(files)
            reader = tf.TFRecordReader()
            key, value = reader.read(fqueue)
            features = tf.parse_single_example(value, features={
                'label': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
            })
            image = tf.image.decode_jpeg(features['image_raw'], channels=3)
            image = tf.cast(image, tf.float32)

            # distort
            image = tf.random_crop(image, [Recognizer.INPUT_SIZE, Recognizer.INPUT_SIZE, 3])
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.4)
            image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
            image = tf.image.random_hue(image, max_delta=0.04)
            image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

            return [tf.image.per_image_whitening(image), features['label']]

        min_queue_examples = num_examples_per_epoch_for_train
        images, labels = tf.train.shuffle_batch_join(
            [read_files(files) for files in queues.values()],
            batch_size=self.batch_size,
            capacity=min_queue_examples + 3 * self.batch_size,
            min_after_dequeue=min_queue_examples
        )
        images = tf.image.resize_images(images, Recognizer.INPUT_SIZE, Recognizer.INPUT_SIZE)
        tf.image_summary('images', images)
        return images, labels

    def inference(self, images, num_classes):
        def _variable_with_weight_decay(name, shape, stddev, wd):
            var = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
            if wd:
                weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            return var

        def _activation_summary(x):
            tensor_name = x.op.name
            tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 3, 40], initializer=tf.truncated_normal_initializer(stddev=0.08))
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[40], initializer=tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 40, 60], initializer=tf.truncated_normal_initializer(stddev=0.08))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[60], initializer=tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 60, 90], initializer=tf.truncated_normal_initializer(stddev=0.08))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[90], initializer=tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv3)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 90, 90], initializer=tf.truncated_normal_initializer(stddev=0.08))
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[90], initializer=tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv4)
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        with tf.variable_scope('fc5') as scope:
            dim = 1
            for d in pool4.get_shape()[1:].as_list():
                dim *= d
            reshape = tf.reshape(pool4, [self.batch_size, dim])
            weights = _variable_with_weight_decay('weights', shape=[dim, 150], stddev=0.02, wd=0.005)
            biases = tf.get_variable('biases', shape=[150], initializer=tf.constant_initializer(0.0))
            fc5 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, weights), biases), name=scope.name)
            _activation_summary(fc5)

        with tf.variable_scope('fc6') as scope:
            weights = _variable_with_weight_decay('weights', shape=[150, 150], stddev=0.02, wd=0.005)
            biases = tf.get_variable('biases', shape=[150], initializer=tf.constant_initializer(0.0))
            fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc5, weights), biases), name=scope.name)
            _activation_summary(fc6)

        with tf.variable_scope('fc7') as scope:
            weights = tf.get_variable('weights', shape=[150, num_classes], initializer=tf.truncated_normal_initializer(stddev=0.02))
            biases = tf.get_variable('biases', shape=[num_classes], initializer=tf.constant_initializer(0.0))
            fc7 = tf.nn.bias_add(tf.matmul(fc6, weights), biases, name=scope.name)
            _activation_summary(fc7)

        return fc7

    def loss(self, logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def train(self, total_loss):
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            tf.scalar_summary(l.op.name + ' (raw)', l)

        # Apply gradients, and add histograms
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer()
            grads = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads)
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables
        variable_averages = tf.train.ExponentialMovingAverage(Recognizer.MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op
