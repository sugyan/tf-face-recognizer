import os
import sys
import tensorflow as tf
import json
from tensorflow.python.framework import graph_util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('datadir', 'data/tfrecords',
                           """Path to the TFRecord data directory.""")
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Path to read model checkpoint.""")
tf.app.flags.DEFINE_string("output_graph", "",
                           """Output 'GraphDef' file name.""")
tf.app.flags.DEFINE_integer('input_size', 96,
                            """Size of input image""")


def main(argv=None):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from model.recognizer import Recognizer

    # read labels data
    labels_path = os.path.join(os.path.join(FLAGS.datadir, 'labels.json'))
    with open(labels_path, 'r') as f:
        labels = json.loads(f.read())
    # create model graph
    r = Recognizer(batch_size=1)
    contents = tf.placeholder(tf.string, name='contents')
    decoded = tf.image.decode_jpeg(contents, channels=3)
    resized = tf.image.resize_images(decoded, [FLAGS.input_size, FLAGS.input_size])
    images = tf.expand_dims(tf.image.per_image_standardization(resized), 0)
    inferences = r.inference(images, len(labels) + 1)
    # restore variables
    variable_averages = tf.train.ExponentialMovingAverage(r.MOVING_AVERAGE_DECAY)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint_path)
        output = graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), [inferences.name.split(':')[0]])
    with open(FLAGS.output_graph, 'wb') as f:
        f.write(output.SerializeToString())


if __name__ == '__main__':
    tf.app.run()
