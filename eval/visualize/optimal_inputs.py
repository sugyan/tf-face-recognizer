import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import tensorflow as tf
from model.recognizer import Recognizer
from eval import inputs

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('target_class', 1,
                           """target class index.""")
tf.app.flags.DEFINE_string('images_dir', 'images',
                           """Directory where to write generated images.""")

def main(argv=None):
    # model definition and train op
    r = Recognizer(batch_size=1)
    # input variable
    with tf.variable_scope('input') as scope:
        image = tf.get_variable('input', shape=(96, 96, 3))
    logits = r.inference(tf.expand_dims(image, 0), FLAGS.num_classes)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, [FLAGS.target_class])
    train_op = tf.train.AdamOptimizer().minimize(losses, var_list=[image])

    output = tf.image.encode_jpeg(tf.image.convert_image_dtype(image + 0.5, tf.uint8, saturate=True))

    saver = tf.train.Saver([v for v in tf.all_variables() if not v.name.startswith('input')])
    checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint)

        tf.train.start_queue_runners(sess=sess)
        for i in range(100):
            _, loss_value = sess.run([train_op, losses])
            print(loss_value)

        filename = 'target-%03d.jpg' % FLAGS.target_class
        with open(os.path.join(os.path.dirname(__file__), '..', '..', FLAGS.images_dir, filename), 'wb') as f:
            f.write(sess.run(output))

if __name__ == '__main__':
    tf.app.run()
