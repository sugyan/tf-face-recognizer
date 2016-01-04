from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf
import numpy as np

cifar10.IMAGE_SIZE = 32
cifar10.NUM_CLASSES = 6
cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 150

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', 'train',
                           """Directory where to read model checkpoints.""")

def eval_once(saver, top_k_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print 'No checkpoint file found'
            return

        tf.train.start_queue_runners(sess=sess)
        predictions = sess.run([top_k_op])
        true_count = np.sum(predictions)
        precision = 1.0 * true_count / FLAGS.batch_size
        print 'precision: %.3f' % precision

def evaluate():
    images, labels = cifar10.inputs(eval_data=True)
    logits = cifar10.inference(images)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = {}
    for v in tf.all_variables():
        if v in tf.trainable_variables():
            restore_name = variable_averages.average_name(v)
        else:
            restore_name = v.op.name
        variables_to_restore[restore_name] = v
    saver = tf.train.Saver(variables_to_restore)
    eval_once(saver, top_k_op)

if __name__ == '__main__':
    evaluate()
