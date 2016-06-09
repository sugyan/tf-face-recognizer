import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.recognizer import Recognizer
import tensorflow as tf
from eval import inputs

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    r = Recognizer()
    filepath = os.path.join(FLAGS.data_dir, 'data-00.tfrecords')
    images, labels = inputs([filepath])
    logits = r.inference(images, FLAGS.num_classes)
    saver = tf.train.Saver(tf.all_variables())

    checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)

        targets = [
            {'name': 'conv1/conv1:0'},
            {'name': 'conv2/conv2:0'},
            {'name': 'conv3/conv3:0'},
            {'name': 'conv4/conv4:0'},
            {'name': 'fc5/fc5:0'},
            {'name': 'fc6/fc6:0'},
            {'name': 'fc7/fc7:0'},
        ]
        for target in targets:
            t = sess.graph.get_tensor_by_name(target['name'])
            print(t)


if __name__ == '__main__':
    tf.app.run()
