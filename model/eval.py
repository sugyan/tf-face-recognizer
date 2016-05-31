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
tf.app.flags.DEFINE_string('eval_dir', 'eval',
                           """Directory where to write event logs.""")

def eval_once(saver, summary_writer, top_k_op, summary_op):
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    for checkpoint_path in ckpt.all_model_checkpoint_paths:
        global_step = checkpoint_path.split('/')[-1].split('-')[-1]

        precisions = []
        for i in range(5):
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
                    precisions.append(precision)

                    if len(precisions) == 5:
                        precisions_avg = sum(precisions) / len(precisions)
                        print 'precision @ %s = %.3f' % (global_step, precisions_avg)

                        summary = tf.Summary()
                        summary.ParseFromString(sess.run(summary_op))
                        summary.value.add(tag='Precision', simple_value=precisions_avg)
                        summary_writer.add_summary(summary, global_step)
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
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir)
    eval_once(saver, summary_writer, top_k_op, summary_op)

if __name__ == '__main__':
    tf.app.run()
