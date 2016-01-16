import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models import v2
from datetime import datetime
import tensorflow as tf
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data/v2/tfrecords',
                           """Path to the TFRecord data directory.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs and checkpoint.""")

def main(argv=None):
    global_step = tf.Variable(0, trainable=False)

    files = [os.path.join(FLAGS.data_dir, 'data%d.tfrecords' % i) for i in range(1, 6)]
    images, labels = v2.inputs(files, distort=True)
    logits = v2.inference(images)
    losses = v2.loss(logits, labels)
    train_op = v2.train(losses, global_step)
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('train', graph_def=sess.graph_def)
        sess.run(tf.initialize_all_variables())

        tf.train.start_queue_runners(sess=sess)

        for step in range(100):
            start_time = time.time()
            _, loss_value = sess.run([train_op, losses])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            format_str = '%s: step %d, loss = %.5f (%.3f sec/batch)'
            print format_str % (datetime.now(), step, loss_value, duration)

            if step % 10 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

if __name__ == '__main__':
    tf.app.run()
