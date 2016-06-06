import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import model
from datetime import datetime
import tensorflow as tf
import numpy as np
import json
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'eval/data/tfrecords',
                           """Path to the TFRecord data directory.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 7001,
                            """Number of batches to run.""")

def main(argv=None):
    filenames = [
        'data-01.tfrecords',
        'data-02.tfrecords',
        'data-03.tfrecords',
        'data-04.tfrecords',
        'data-05.tfrecords',
    ]
    files = [os.path.join(FLAGS.data_dir, f) for f in filenames]
    images, labels = model.inputs(files, distort=True)
    logits = model.inference(images, 40)
    losses = model.loss(logits, labels)
    train_op = model.train(losses)
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=36)
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('train', graph=sess.graph)
        sess.run(tf.initialize_all_variables())

        tf.train.start_queue_runners(sess=sess)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, losses])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            format_str = '%s: step %d, loss = %.5f (%.3f sec/batch)'
            print(format_str % (datetime.now(), step, loss_value, duration))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % 200 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

if __name__ == '__main__':
    tf.app.run()
