from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf
import time
import os

cifar10.IMAGE_SIZE = 48
cifar10.NUM_CLASSES = 6
# cifar10.FLAGS.batch_size = 64
cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
# cifar10.NUM_EPOCHS_PER_DECAY = 1000.0
# cifar10.LEARNING_RATE_DECAY_FACTOR = 0.02
# cifar10.INITIAL_LEARNING_RATE = 0.05

def inputs():
    # target files
    filenames = []
    with open('./data/filelist.txt', 'r') as f:
        for line in f:
            filenames.append(line.strip())
    fqueue = tf.train.string_input_producer(filenames)

    # read and parse from file queue
    height = 96
    width = 96
    depth = 3
    label_bytes = 1
    image_bytes = height * width * depth
    record_bytes = image_bytes + label_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(fqueue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    # The first bytes represent the label, which we convert from uint8->int32.
    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    # The remaining bytes after the label represent the image, which we reshape to [height, width, depth].
    uint8image = tf.reshape(
        tf.slice(record_bytes, [label_bytes], [image_bytes]),
        [height, width, depth]
    )
    image = tf.cast(uint8image, tf.float32)
    return image, label

def train():
    checkpoint_dir = 'train'
    checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')

    image, label = inputs()
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size = cifar10.FLAGS.batch_size,
        capacity = 3 * cifar10.FLAGS.batch_size,
        min_after_dequeue = 1
    )
    labels = tf.reshape(label_batch, [cifar10.FLAGS.batch_size])

    logits = cifar10.inference(tf.image.resize_images(images, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE))
    loss = cifar10.loss(logits, labels)
    global_step = tf.Variable(0, trainable=False)
    train_op = cifar10.train(loss, global_step)
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
        prev_ckpt = None
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            prev_ckpt = ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            init = tf.initialize_all_variables()
            sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            loss_values = []
            for i in range(100):
                start_time = time.time()
                _, loss_value, global_step_value = sess.run([train_op, loss, global_step])
                if loss_value > 1e6:
                    break
                loss_values.append(loss_value)
                duration = time.time() - start_time
                print '%3d: %f (%.3f sec/batch)' % (global_step_value, loss_value, duration)
            if loss_values[0] > loss_values[-1]:
                if prev_ckpt is not None:
                    os.remove(prev_ckpt)
                saver.save(sess, checkpoint_path, global_step=global_step_value)
            else:
                print 'train failed!'
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads)

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
