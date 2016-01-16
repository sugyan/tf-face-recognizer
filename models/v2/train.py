import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.v2 import main as v2
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data/v2/tfrecords',
                           """Path to the TFRecord data directory.""")

def main(argv=None):
    global_step = tf.Variable(0, trainable=False)
    images, labels = v2.inputs(FLAGS.data_dir, distort=True)
    logits = v2.inference(images)
    losses = v2.loss(logits, labels)
    train_op = v2.train(losses, global_step)
    # saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('train', graph_def=sess.graph_def)
        sess.run(tf.initialize_all_variables())

        tf.train.start_queue_runners(sess=sess)

        for step in range(20):
            _, loss_value = sess.run([train_op, losses])
            print loss_value

            if step % 4 == 0:
                summary = sess.run(tf.merge_all_summaries())
                summary_writer.add_summary(summary)

if __name__ == '__main__':
    tf.app.run()
