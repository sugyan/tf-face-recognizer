from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf

cifar10.IMAGE_SIZE = 96
cifar10.NUM_CLASSES = 6
cifar10.FLAGS.batch_size = 1

label = tf.cast(tf.slice(tf.decode_raw("\x01", tf.uint8), [0], [1]), tf.int32)

sess = tf.Session()
with sess.as_default():
    image = tf.image.decode_jpeg(open('./image', 'r').read()).eval()
image = tf.cast(image, tf.float32)

images, label_batch = tf.train.shuffle_batch(
    [image, label],
    batch_size = cifar10.FLAGS.batch_size,
    capacity = 3 * cifar10.FLAGS.batch_size,
    min_after_dequeue = 1
)
labels = tf.reshape(label_batch, [cifar10.FLAGS.batch_size])

# images, labels = cifar10.inputs(eval_data='test')
# print images

logits = cifar10.inference(images)
for variable in tf.all_variables():
    print variable.name

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        # while not coord.should_stop():
            # Run training steps or whatever
        print sess.run(logits)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    coord.request_stop()
    coord.join(threads)
