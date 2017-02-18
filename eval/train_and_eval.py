import os
import math
import sys
import time
from datetime import datetime

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('datadir', os.path.join(os.path.dirname(__file__), 'data', 'tfrecords'),
                           """Path to the TFRecord data directory.""")
tf.app.flags.DEFINE_string('eval_file',
                           os.path.join(os.path.dirname(__file__), 'data', 'tfrecords', 'data-00.tfrecords'),
                           """Path to the TFRecord for evaluation.""")
tf.app.flags.DEFINE_string('logdir',
                           os.path.join(os.path.dirname(__file__), 'logdir'),
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 19200,
                            'Number of examples for train')
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_eval',   4800,
                            'Number of examples for evaluation')
tf.app.flags.DEFINE_integer('max_steps', 20001,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_classes', 120,
                            """Number of classes.""")


def distorted_inputs(filenames, distortion=0, batch_size=128):
    fqueue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, value = reader.read(fqueue)
    features = tf.parse_single_example(value, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })
    label = features['label']
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)

    if distortion == 0:
        image = tf.random_crop(image, [96, 96, 3])
    if distortion == 1:
        bounding_boxes = tf.div(tf.constant([[[8, 8, 104, 104]]], dtype=tf.float32), 112.0)
        begin, size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image), bounding_boxes,
            min_object_covered=(80.0*80.0)/(96.0*96.0),
            aspect_ratio_range=[9.0/10.0, 10.0/9.0])
        image = tf.slice(image, begin, size)
        image = tf.image.resize_images(image, [96, 96])
    # common distortion
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_hue(image, max_delta=0.04)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
    image = tf.image.per_image_standardization(image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(FLAGS.num_examples_per_epoch_for_train * min_fraction_of_examples_in_queue)
    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size, min_queue_examples + 3 * batch_size, min_queue_examples)
    tf.summary.image('disotrted_inputs', images, max_outputs=16)
    return images, labels


def inputs(filename, batch_size=100):
    fqueue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, value = reader.read(fqueue)
    features = tf.parse_single_example(value, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })
    label = features['label']
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 96, 96)
    image = tf.image.per_image_standardization(image)

    images, labels = tf.train.batch([image, label], batch_size)
    tf.summary.image('inputs', images, max_outputs=16)
    return images, labels


def main(argv=None):
    filenames = []
    for f in [x for x in os.listdir(FLAGS.datadir) if x.endswith('.tfrecords')]:
        filepath = os.path.join(FLAGS.datadir, f)
        if filepath != FLAGS.eval_file:
            filenames.append(filepath)
    t_images, t_labels = distorted_inputs(filenames, distortion=1)
    e_images, e_labels = inputs(FLAGS.eval_file)

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import model
    t_logits = model.inference(t_images, FLAGS.num_classes, reuse=False)
    e_logits = model.inference(e_images, FLAGS.num_classes, reuse=True)
    # train ops
    losses = model.loss(t_logits, t_labels)
    train_op = model.train(losses)
    is_nan = tf.is_nan(losses)
    # eval ops, variables
    e_batch_size = int(e_logits.get_shape()[0])
    num_iter = int(math.ceil(1.0 * FLAGS.num_examples_per_epoch_for_eval / e_batch_size))
    true_count_op = tf.reduce_sum(tf.train.batch([tf.count_nonzero(tf.nn.in_top_k(e_logits, e_labels, 1))], num_iter))
    total_count = num_iter * e_batch_size

    # summary
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        # initialize (and restore) variables
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            print('restore variables from {}.'.format(ckpt.model_checkpoint_path))
            tf.train.Saver(tf.trainable_variables()).restore(sess, ckpt.model_checkpoint_path)
        # checkpoint saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=21)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value, is_nan_value = sess.run([train_op, losses, is_nan])
            duration = time.time() - start_time

            assert not is_nan_value, 'Model diverged with loss = NaN'

            print('{}: step {:05d}, loss = {:.5f} ({:.3f} sec/batch)'.format(
                datetime.now(), step, loss_value, duration))

            if step % 500 == 0:
                true_count = sess.run(true_count_op)
                precision = 100.0 * true_count / total_count
                print('{}: precision = {:.3f} %'.format(datetime.now(), precision))
                # write summary
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='precision', simple_value=precision)
                summary_writer.add_summary(summary, global_step=step)
            if step % 1000 == 0:
                checkpoint_path = os.path.join(FLAGS.logdir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
