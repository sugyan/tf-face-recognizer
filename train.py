from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf
import time
import os

cifar10.IMAGE_SIZE = 48
cifar10.NUM_CLASSES = 6
cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 250
cifar10.NUM_EPOCHS_PER_DECAY = 1000.0
# cifar10.INITIAL_LEARNING_RATE = 0.09

def inputs():
    # read from target files
    filenames = []
    with open('./data/filelist.txt', 'r') as f:
        for line in f:
            filenames.append(line.strip())
    fqueue = tf.train.string_input_producer(filenames)

    label_bytes = 1
    image_bytes = 48 * 48 * 3
    record_bytes = image_bytes + label_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(fqueue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    # The first bytes represent the label, which we convert from uint8->int32.
    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    # The remaining bytes after the label represent the image, which we reshape to [height, width, depth].
    uint8image = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [3, 48, 48])
    image = tf.transpose(tf.cast(uint8image, tf.float32), [1, 2, 0])
    return tf.train.shuffle_batch(
        [image, label],
        batch_size = cifar10.FLAGS.batch_size,
        capacity = 3 * cifar10.FLAGS.batch_size,
        min_after_dequeue = 1
    )

def train(checkpoint_dir):
    # ops
    global_step = tf.Variable(0, trainable=False)
    images, labels = inputs()
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
            if global_step_value % 1000 == 0:
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
