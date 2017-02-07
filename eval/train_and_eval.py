import json
import os
import time

from datetime import datetime
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('datadir', os.path.join(os.path.dirname(__file__), 'data', 'tfrecords'),
                           """Path to the TFRecord data directory.""")
tf.app.flags.DEFINE_string('eval_file',
                           os.path.join(os.path.dirname(__file__), 'data', 'tfrecords', 'data-00.tfrecords'),
                           """Path to the TFRecord for evaluation.""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 19_200,
                            'number of examples')
tf.app.flags.DEFINE_string('logdir',
                           os.path.join(os.path.dirname(__file__), 'logdir'),
                           """Directory where to write event logs and checkpoint.""")
# tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
#                            """Path to read model checkpoint.""")
# tf.app.flags.DEFINE_integer('input_size', 96,
#                             """Size of input image""")
# tf.app.flags.DEFINE_integer('max_steps', 5001,
#                             """Number of batches to run.""")


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
            min_object_covered=0.9)
        image = tf.slice(image, begin, size)
        image = tf.image.resize_images(image, tf.to_int32(tf.truncated_normal([2], mean=96.0, stddev=24.0)))
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


# def restore_or_initialize(sess):
#     for v in tf.global_variables():
#         if v in tf.trainable_variables() or "ExponentialMovingAverage" in v.name:
#             try:
#                 print('restore variable "%s"' % v.name)
#                 restorer = tf.train.Saver([v])
#                 restorer.restore(sess, FLAGS.checkpoint_path)
#             except Exception:
#                 print('could not restore, initialize!')
#                 sess.run(tf.variables_initializer([v]))
#         else:
#             print('initialize variable "%s"' % v.name)
#             sess.run(tf.variables_initializer([v]))


def main(argv=None):
    filenames = []
    for f in [x for x in os.listdir(FLAGS.datadir) if x.endswith('.tfrecords')]:
        filepath = os.path.join(FLAGS.datadir, f)
        if filepath != FLAGS.eval_file:
            filenames.append(filepath)
    t_images, t_labels = distorted_inputs(filenames, distortion=1)
    e_images, e_labels = inputs(FLAGS.eval_file)
    # logits = model.inference(images, len(json.loads(labels_data)) + 1)
    # losses = model.loss(logits, labels)
    # train_op = model.train(losses)
    summary_op = tf.summary.merge_all()
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=21)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        #     restore_or_initialize(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print(sess.run([e_images, e_labels]))
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str)

    #     for step in range(FLAGS.max_steps):
    #         start_time = time.time()
    #         _, loss_value = sess.run([train_op, losses])
    #         duration = time.time() - start_time

    #         assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    #         format_str = '%s: step %d, loss = %.5f (%.3f sec/batch)'
    #         print(format_str % (datetime.now(), step, loss_value, duration))

    #         if step % 100 == 0:
    #             summary_str = sess.run(summary_op)
    #             summary_writer.add_summary(summary_str, step)
    #         if step % 250 == 0 or (step + 1) == FLAGS.max_steps:
    #             checkpoint_path = os.path.join(FLAGS.logdir, 'model.ckpt')
    #             saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False, write_state=False)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
