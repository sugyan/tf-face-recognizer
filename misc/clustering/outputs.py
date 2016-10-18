import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import tensorflow as tf
from model.recognizer import Recognizer

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Directory where to read model checkpoints.""")
INPUT_IMAGE_SIZE = 96

r = Recognizer(batch_size=1)

with tf.Session() as sess:
    labels = tf.Variable(tf.bytes(), name='labels', trainable=False)
    tf.train.Saver([labels]).restore(sess, FLAGS.checkpoint_path)
    num_classes = len(json.loads(sess.run(labels).decode())) + 1

def main(argv=None):
    data = tf.placeholder(tf.string)
    image = tf.image.decode_jpeg(data, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    inputs = tf.expand_dims(tf.image.per_image_whitening(image), 0)
    r.inference(inputs, num_classes)
    fc5 = tf.get_default_graph().get_tensor_by_name('fc5/fc5:0')
    fc6 = tf.get_default_graph().get_tensor_by_name('fc6/fc6:0')

    outputs = {}
    with tf.Session() as sess:
        variable_averages = tf.train.ExponentialMovingAverage(r.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, FLAGS.checkpoint_path)

        dirname = os.path.join(os.path.dirname(__file__), 'images')
        for filename in os.listdir(dirname):
            if not filename.endswith('.jpg'):
                continue
            with open(os.path.join(dirname, filename), 'rb') as f:
                fc5out, fc6out = sess.run([fc5, fc6], feed_dict={data: f.read()})
            outputs[filename] = {
                'fc5': fc5out.flatten().tolist(),
                'fc6': fc6out.flatten().tolist(),
            }
    for out in ['fc5', 'fc6']:
        filename = os.path.join(os.path.dirname(__file__), '%s.csv' % out)
        with open(filename, 'w') as f:
            for name, values in outputs.items():
                f.write(','.join([name] + [str(x) for x in values[out]]) + '\n')

if __name__ == '__main__':
    tf.app.run()
