import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.recognizer import Recognizer
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'eval/data/tfrecords',
                           """Path to the TFRecord data directory.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('num_classes', 40,
                            """number of class""")

def inputs(files, batch_size=128, original_image_size=112, cropped_image_size=96, image_size=96):
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
    image.set_shape([original_image_size, original_image_size, 3])
    image = tf.image.resize_image_with_crop_or_pad(image, cropped_image_size, cropped_image_size)
    images, labels = tf.train.batch(
        [tf.image.per_image_whitening(image), features['label']], batch_size
    )
    return tf.image.resize_images(images, image_size, image_size), labels

def main(argv=None):
    r = Recognizer(batch_size=100)
    filepath = os.path.join(FLAGS.data_dir, 'data-00.tfrecords')
    images, labels = inputs([filepath], batch_size=r.batch_size)
    logits = r.inference(images, FLAGS.num_classes)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    saver = tf.train.Saver(tf.all_variables())

    num_samples = 1200
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)

        state = tf.train.get_checkpoint_state(FLAGS.train_dir)
        for checkpoint in state.all_model_checkpoint_paths:
            print('restore %s' % checkpoint)
            saver.restore(sess, checkpoint)

            true_count = 0
            for i in range(num_samples / 100):
                predictions = sess.run(top_k_op)
                true_count += np.sum(predictions)
            precision = float(true_count) / num_samples
            print('precision: %.5f' % precision)


if __name__ == '__main__':
    tf.app.run()
