import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.recognizer import Recognizer
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('input_file', '',
                           """Path to the TFRecord data.""")
tf.app.flags.DEFINE_string('logdir', os.path.join(os.path.dirname(__file__), 'logdir'),
                           """Directory where to write checkpoints.""")


def inputs(files, batch_size=0):
    fqueue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    key, value = reader.read(fqueue)
    features = tf.parse_single_example(value, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
    })
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 96, 96)
    return tf.train.batch(
        [tf.image.per_image_standardization(image)], batch_size
    )


def main(argv=None):
    filepath = FLAGS.input_file
    if filepath and not os.path.exists(filepath):
        raise Exception('%s does not exist' % filepath)

    r = Recognizer(batch_size=200)
    images = inputs([filepath], batch_size=r.batch_size)
    r.inference(images, 1)
    fc5 = tf.get_default_graph().get_tensor_by_name('fc5/fc5:0')
    fc6 = tf.get_default_graph().get_tensor_by_name('fc6/fc6:0')
    with tf.Session() as sess:
        variable_averages = tf.train.ExponentialMovingAverage(r.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        for name, v in variables_to_restore.items():
            try:
                tf.train.Saver([v]).restore(sess, FLAGS.checkpoint_path)
            except Exception:
                sess.run(tf.variables_initializer([v]))

        tf.train.start_queue_runners(sess=sess)
        outputs = sess.run({'fc5': fc5, 'fc6': fc6})

        targets = [tf.Variable(e, name=name) for name, e in outputs.items()]
        sess.run(tf.variables_initializer(targets))
        graph_saver = tf.train.Saver(targets)
        graph_saver.save(sess, os.path.join(FLAGS.logdir, 'model.ckpt'))


if __name__ == '__main__':
    tf.app.run()
