import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.recognizer import Recognizer
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
tf.app.flags.DEFINE_integer('max_steps', 8001,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_classes', 40,
                            """number of class""")

def main(argv=None):
    r = Recognizer()
    filenames = [
        'data-01.tfrecords',
        'data-02.tfrecords',
        'data-03.tfrecords',
        'data-04.tfrecords',
        'data-05.tfrecords',
    ]
    files = [os.path.join(FLAGS.data_dir, f) for f in filenames]
    images, labels = r.inputs(files, num_examples_per_epoch_for_train=6000)
    logits = r.inference(images, FLAGS.num_classes)
    losses = r.loss(logits, labels)
    train_op = r.train(losses)
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=41)
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.train.SummaryWriter('train', graph=sess.graph)

        sess.run(tf.initialize_all_variables())
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
