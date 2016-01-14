import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data/v2/tfrecords',
                           """Path to the TFRecord data directory.""")

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '128'))
NUM_CLASSES = int(os.environ.get('NUM_CLASSES', '10'))

def distorted_inputs(data_dir):
    filenames = [os.path.join(data_dir, 'data%d.tfrecords' % i) for i in range(1, 3)]
    fqueue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    key, value = reader.read(fqueue)
    features = tf.parse_single_example(value, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })
    jpeg = tf.image.decode_jpeg(features['image_raw'], channels=3)
    image = tf.cast(jpeg, tf.float32)
    cropped = tf.image.random_crop(image, [96, 96])
    return tf.train.shuffle_batch([cropped, features['label']], batch_size=BATCH_SIZE, capacity=2, min_after_dequeue=1)

def inference(images):
    def _variable_with_weight_decay(name, shape, stddev, wd):
        var = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 3, 32], initializer=tf.truncated_normal_initializer(stddev=1e-4))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=tf.truncated_normal_initializer(stddev=1e-4))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=1e-4))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[128], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('conv4') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=1e-4))
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    with tf.variable_scope('fc5') as scope:
        dim = 1
        for d in pool4.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool4, [BATCH_SIZE, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024], stddev=0.05, wd=0.005)
        biases = tf.get_variable('biases', shape=[1024], initializer=tf.constant_initializer(0.1))
    fc5 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)

    with tf.variable_scope('fc6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, 256], stddev=0.05, wd=0.005)
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.1))
    fc6 = tf.nn.relu_layer(fc5, weights, biases, name=scope.name)

    with tf.variable_scope('fc7') as scope:
        weights = _variable_with_weight_decay('weights', shape=[256, NUM_CLASSES], stddev=0.05, wd=0.005)
        biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.1))
    fc7 = tf.nn.xw_plus_b(fc6, weights, biases, name=scope.name)

    return fc7

def loss(logits, labels):
    # TODO
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def main(argv=None):
    images, labels = distorted_inputs(FLAGS.data_dir)
    logits = inference(images)
    print logits
    # losses = loss(logits, labels)
    # saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        # summary_writer = tf.train.SummaryWriter('train', graph_def=sess.graph_def)
        sess.run(tf.initialize_all_variables())

        tf.train.start_queue_runners(sess=sess)
        print sess.run(logits)

if __name__ == '__main__':
    tf.app.run()
