from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf

cifar10.IMAGE_SIZE = 96
cifar10.NUM_CLASSES = 6
cifar10.FLAGS.batch_size = 10

def inputs():
    # target files
    filenames = []
    with open('./data/filelist.txt', 'r') as f:
        for line in f:
            filenames.append(line.strip())
    fqueue = tf.train.string_input_producer(filenames)

    # read and parse from file queue
    depth = 3
    label_bytes = 1
    image_bytes = cifar10.IMAGE_SIZE * cifar10.IMAGE_SIZE * depth
    record_bytes = image_bytes + label_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(fqueue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    # The first bytes represent the label, which we convert from uint8->int32.
    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    # The remaining bytes after the label represent the image, which we reshape to [height, width, depth].
    uint8image = tf.reshape(
        tf.slice(record_bytes, [label_bytes], [image_bytes]),
        [cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, depth]
    )
    image = tf.cast(uint8image, tf.float32)
    return image, label

def train():
    image, label = inputs()
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size = cifar10.FLAGS.batch_size,
        capacity = 3 * cifar10.FLAGS.batch_size,
        min_after_dequeue = 1
    )
    labels = tf.reshape(label_batch, [cifar10.FLAGS.batch_size])
    logits = cifar10.inference(images)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            print sess.run(logits)
            # TODO
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train()
