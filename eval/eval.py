import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ['BATCH_SIZE'] = '100'

import model
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'eval/data/tfrecords',
                           """Path to the TFRecord data directory.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('image_size', 96,
                            """size of input image""")
tf.app.flags.DEFINE_integer('num_classes', 40,
                            """number of class""")

ORIGINAL_IMAGE_SIZE = 112
CROPPED_IMAGE_SIZE = 96

def inputs(files):
    fqueue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    key, value = reader.read(fqueue)
    features = tf.parse_single_example(value, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    shape = tf.shape(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE, 3])
    image = tf.image.resize_image_with_crop_or_pad(image, CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE)
    images, labels = tf.train.batch(
        [tf.image.per_image_whitening(image), features['label']], 100
    )
    return tf.image.resize_images(images, FLAGS.image_size, FLAGS.image_size), labels


def main(argv=None):
    filepath = os.path.join(FLAGS.data_dir, 'data-00.tfrecords')
    images, labels = inputs([filepath])
    logits = model.inference(images, FLAGS.num_classes)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)

        state = tf.train.get_checkpoint_state(FLAGS.train_dir)
        for checkpoint in state.all_model_checkpoint_paths:
            print('restore %s' % checkpoint)
            saver.restore(sess, checkpoint)

            true_count = 0
            # 100 x 12 = 1200
            for i in range(12):
                predictions = sess.run(top_k_op)
                true_count += np.sum(predictions)
            precision = float(true_count) / 1200
            print('precision: %.5f' % precision)


if __name__ == '__main__':
    tf.app.run()
