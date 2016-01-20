import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models import v2
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data/v2/tfrecords',
                           """Path to the TFRecord data directory.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', 'eval_dir',
                           """Directory where to write event logs.""")

def eval_once(saver, top_k_op):
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    for checkpoint_path in ckpt.all_model_checkpoint_paths:
        global_step = checkpoint_path.split('/')[-1].split('-')[-1]
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(FLAGS.train_dir, checkpoint_path))

            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                total_sample_count = v2.BATCH_SIZE
                true_count = 0
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                precision = 1.0 * true_count / total_sample_count
                print('precision @ %s = %.3f' % (global_step, precision))
            except Exception as e:
                coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):
    files = [os.path.join(FLAGS.data_dir, 'test.tfrecords')]
    images, labels = v2.inputs(files, distort=False)
    logits = v2.inference(images)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    variable_averages = tf.train.ExponentialMovingAverage(v2.MOVING_AVERAGE_DECAY)
    variables_to_restore = {}
    for v in tf.all_variables():
        if v in tf.trainable_variables():
            name = variable_averages.average_name(v)
        else:
            name = v.op.name
        variables_to_restore[name] = v
    saver = tf.train.Saver(variables_to_restore)

    eval_once(saver, top_k_op)

if __name__ == '__main__':
    tf.app.run()
