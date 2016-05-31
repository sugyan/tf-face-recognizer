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

tf.app.flags.DEFINE_string('data_dir', 'data/tfrecords',
                           """Path to the TFRecord data directory.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('max_steps', 5001,
                            """Number of batches to run.""")

def labels_json():
    filepath = os.path.join(os.path.join(FLAGS.data_dir, 'labels.json'))
    with open(filepath, 'r') as f:
        return f.read()

def restore_or_initialize(sess):
    if os.path.exists(FLAGS.checkpoint_path):
        for v in tf.all_variables():
            if v in tf.trainable_variables() or "ExponentialMovingAverage" in v.name:
                try:
                    print 'restore variable "%s"' % v.name
                    restorer = tf.train.Saver([v])
                    restorer.restore(sess, FLAGS.checkpoint_path)
                except Exception:
                    print 'could not restore, initialize!'
                    sess.run(tf.initialize_variables([v]))
            else:
                print 'initialize variable "%s"' % v.name
                sess.run(tf.initialize_variables([v]))
    else:
        print 'initialize all variables'
        sess.run(tf.initialize_all_variables())

def main(argv=None):
    labels_data = labels_json()
    tf.Variable(labels_data, trainable=False, name='labels')

    files = [os.path.join(FLAGS.data_dir, f) for f in os.listdir(os.path.join(FLAGS.data_dir)) if f.endswith('.tfrecords')]
    images, labels = model.inputs(files, distort=True)
    logits = model.inference(images, len(json.loads(labels_data)) + 1)
    losses = model.loss(logits, labels)
    train_op = model.train(losses)
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=21)
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('train', graph=sess.graph)
        restore_or_initialize(sess)

        tf.train.start_queue_runners(sess=sess)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, losses])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            format_str = '%s: step %d, loss = %.5f (%.3f sec/batch)'
            print format_str % (datetime.now(), step, loss_value, duration)

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % 250 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

if __name__ == '__main__':
    tf.app.run()
