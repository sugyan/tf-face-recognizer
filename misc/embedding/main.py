import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.recognizer import Recognizer

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_file', 'data.tfrecords',
                           """Path to the TFRecord data.""")
tf.app.flags.DEFINE_string('logdir', os.path.join(os.path.dirname(__file__), 'logdir'),
                           """Directory where to write checkpoints.""")

def inputs(files, batch_size=0):
    fqueue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    key, value = reader.read(fqueue)
    features = tf.parse_single_example(value, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 96, 96)
    return tf.train.batch(
        [tf.image.per_image_standardization(image), features['label']], batch_size
    )

def main(argv=None):
    filepath = FLAGS.input_file
    if not os.path.exists(filepath):
        raise Exception('%s does not exist' % filepath)

    num_classes = 10 # TODO
    r = Recognizer(batch_size=1)
    images, labels = inputs([filepath], batch_size=r.batch_size)
    logits = r.inference(images, num_classes)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join(FLAGS.logdir, "model.ckpt"))

if __name__ == '__main__':
    tf.app.run()
