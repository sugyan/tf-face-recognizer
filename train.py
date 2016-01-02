from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf
import time
import os

cifar10.NUM_CLASSES = 6
cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 250

def train(checkpoint_dir):
    # ops
    global_step = tf.Variable(0, trainable=False)
    images, labels = cifar10.inputs(False)
    logits = cifar10.inference(tf.image.resize_images(images, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE))
    loss = cifar10.loss(logits, labels)
    train_op = cifar10.train(loss, global_step)
    summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
        summary_writer = tf.train.SummaryWriter(checkpoint_dir, graph_def=sess.graph_def)
        if not tf.train.get_checkpoint_state(checkpoint_dir):
            sess.run(tf.initialize_all_variables())
            saver.save(sess, checkpoint_path, global_step=0)

        tf.train.start_queue_runners(sess=sess)
        while True:
            # restore or initialize variables
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            loss_values = []
            # train
            for step in range(100):
                start_time = time.time()
                _, loss_value, global_step_value = sess.run([train_op, loss, global_step])
                duration = time.time() - start_time
                print '%d: %f (%.3f sec/batch)' % (global_step_value, loss_value, duration)
                loss_values.append(loss_value)
                if loss_value > 1e6:
                    break
            if not loss_values[0] > loss_values[-1]:
                print 'failed.'
                continue
            # save and remove old checkpoint
            saver.save(sess, checkpoint_path, global_step=global_step_value)
            os.remove(ckpt.model_checkpoint_path)
            # add summary
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, global_step_value)

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
    train(os.path.join(os.path.dirname(__file__), 'train'))
