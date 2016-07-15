import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import tensorflow as tf
from model.recognizer import Recognizer

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('images_dir', 'eval/images',
                           """Directory where to read images.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('num_classes', 40,
                            """number of class""")

def main(argv=None):
    # setup recognizer
    r = Recognizer(batch_size=1)
    p = tf.placeholder(tf.string)
    decoded = tf.image.decode_jpeg(p, channels=3)
    resized = tf.image.resize_images(decoded, r.INPUT_SIZE, r.INPUT_SIZE)
    inputs = tf.expand_dims(tf.image.per_image_whitening(resized), 0)
    fc7 = r.inference(inputs, FLAGS.num_classes)
    # restore variables
    variable_averages = tf.train.ExponentialMovingAverage(r.MOVING_AVERAGE_DECAY)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    rows = []
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
        saver.restore(sess, checkpoint)

        fc5 = sess.graph.get_operation_by_name('fc5/fc5').outputs[0]
        fc6 = sess.graph.get_operation_by_name('fc6/fc6').outputs[0]
        for filename in [f for f in os.listdir(FLAGS.images_dir) if f.endswith('.jpg')]:
            filepath = os.path.join(FLAGS.images_dir, filename)
            with open(filepath, 'rb') as f:
                data = f.read()
            label = filename.split('-')[0]
            outputs = sess.run([fc5, fc6, fc7], feed_dict={p: data})
            rows.append({
                'label': label,
                'fc7': outputs[0].flatten().tolist(),
                'fc6': outputs[1].flatten().tolist(),
                'fc5': outputs[2].flatten().tolist(),
            })
    # write to file in svmlight / libsvm format
    for target in ['fc5', 'fc6', 'fc7']:
        with open('%s.txt' % target, 'w') as f:
            for row in rows:
                f.write(row['label'] + ' ')
                f.write(" ".join(["%d:%f" % (i, row[target][i]) for i in range(len(row[target]))]))
                f.write('\n')

if __name__ == '__main__':
    tf.app.run()
