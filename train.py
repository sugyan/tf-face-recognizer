from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf
import numpy as np
import time
import os

cifar10.IMAGE_SIZE = 32
cifar10.NUM_CLASSES = 6
cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 750
cifar10.NUM_EPOCHS_PER_DECAY = 500.0
cifar10.LEARNING_RATE_DECAY_FACTOR = 0.3

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

def train():
    # ops
    global_step = tf.Variable(0, trainable=False)
    images, labels = cifar10.distorted_inputs()
    logits = cifar10.inference(tf.image.resize_images(images, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE))
    loss = cifar10.loss(logits, labels)
    train_op = cifar10.train(loss, global_step)
    summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)

        # restore or initialize variables
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.initialize_all_variables())

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        start = sess.run(global_step)
        for step in xrange(start, FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            print '%d: %f (%.3f sec/batch)' % (step, loss_value, duration)

            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                # Save the model checkpoint periodically.
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def print_parameters():
    print '''
    FLAGS.batch_size = %s
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = %s
    NUM_EPOCHS_PER_DECAY = %s
    LEARNING_RATE_DECAY_FACTOR = %s
    INITIAL_LEARNING_RATE = %s
    ''' % (
        cifar10.FLAGS.batch_size,
        cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
        cifar10.NUM_EPOCHS_PER_DECAY,
        cifar10.LEARNING_RATE_DECAY_FACTOR,
        cifar10.INITIAL_LEARNING_RATE,
    )

if __name__ == '__main__':
    print_parameters()
    train()
