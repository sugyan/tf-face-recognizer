import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.recognizer import Recognizer
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('imgdir', os.path.join(os.path.dirname(__file__), 'images'),
                           """Path to the images directory.""")
tf.app.flags.DEFINE_string('logdir', os.path.join(os.path.dirname(__file__), 'logdir'),
                           """Directory where to write checkpoints.""")


def main(argv=None):
    if not os.path.exists(FLAGS.imgdir):
        raise Exception('%s does not exist' % FLAGS.imgdir)

    r = Recognizer(batch_size=1)
    data = tf.placeholder(tf.string)
    orig_image = tf.image.decode_jpeg(data, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(orig_image, 96, 96)
    image = tf.image.per_image_standardization(image)
    r.inference(tf.expand_dims(image, axis=0), 1)
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

        for file in os.listdir(FLAGS.imgdir):
            with open(os.path.join(FLAGS.imgdir, file), 'rb') as f:
                outputs = sess.run({
                    'fc5': fc5,
                    'fc6': fc6,
                    'image': orig_image
                }, feed_dict={data: f.read()})
            print(outputs)


if __name__ == '__main__':
    tf.app.run()
