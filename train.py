from datetime import datetime
import tensorflow as tf
import numpy as np
import model

import json
import os
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('datadir', 'data/tfrecords',
                           """Path to the TFRecord data directory.""")
tf.app.flags.DEFINE_string('logdir', 'logdir',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Path to read model checkpoint.""")
tf.app.flags.DEFINE_integer('input_size', 96,
                            """Size of input image""")
tf.app.flags.DEFINE_integer('max_steps', 5001,
                            """Number of batches to run.""")


def inputs(batch_size, files, num_examples_per_epoch_for_train=5000):
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
        bounding_boxes = tf.div(tf.constant([[[8, 8, 104, 104]]], dtype=tf.float32), 112.0)
        begin, size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image), bounding_boxes,
            min_object_covered=(80.0*80.0)/(96.0*96.0),
            aspect_ratio_range=[9.0/10.0, 10.0/9.0])
        image = tf.slice(image, begin, size)
        image = tf.image.resize_images(image, [FLAGS.input_size, FLAGS.input_size])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_hue(image, max_delta=0.04)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

        return [tf.image.per_image_standardization(image), features['label']]

    min_queue_examples = num_examples_per_epoch_for_train
    images, labels = tf.train.shuffle_batch_join(
        [read_files(files) for files in queues.values()],
        batch_size=batch_size,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples
    )
    images = tf.image.resize_images(images, [FLAGS.input_size, FLAGS.input_size])
    tf.summary.image('images', images)
    return images, labels


def labels_json():
    filepath = os.path.join(os.path.join(FLAGS.datadir, 'labels.json'))
    with open(filepath, 'r') as f:
        return f.read()


def restore_or_initialize(sess):
    for v in tf.global_variables():
        if v in tf.trainable_variables() or "ExponentialMovingAverage" in v.name:
            try:
                print('restore variable "%s"' % v.name)
                restorer = tf.train.Saver([v])
                restorer.restore(sess, FLAGS.checkpoint_path)
            except Exception:
                print('could not restore, initialize!')
                sess.run(tf.variables_initializer([v]))
        else:
            print('initialize variable "%s"' % v.name)
            sess.run(tf.variables_initializer([v]))


def main(argv=None):
    labels_data = labels_json()
    tf.Variable(labels_data, trainable=False, name='labels')

    batch_size = 128
    files = [os.path.join(FLAGS.datadir, f) for f in os.listdir(os.path.join(FLAGS.datadir)) if f.endswith('.tfrecords')]
    images, labels = inputs(batch_size, files)
    logits = model.inference(images, len(json.loads(labels_data)) + 1)
    losses = model.loss(logits, labels)
    train_op = model.train(losses)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=21)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph=sess.graph)
        restore_or_initialize(sess)

        tf.train.start_queue_runners(sess=sess)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, losses])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            format_str = '%s: step %d, loss = %.5f (%.3f sec/batch)'
            print(format_str % (datetime.now(), step, loss_value, duration))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % 250 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.logdir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False, write_state=False)


if __name__ == '__main__':
    tf.app.run()
