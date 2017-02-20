import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_file',
                           os.path.join(os.path.dirname(__file__), '..', 'data', 'tfrecords', 'data-00.tfrecords'),
                           """Path to the TFRecord.""")


def main(argv=None):
    example = tf.placeholder(tf.string)
    features = tf.parse_single_example(example, features={
        'image_raw': tf.FixedLenFeature([], tf.string)
    })
    decode = tf.image.decode_jpeg(features['image_raw'], channels=3)

    size = 16
    with tf.Session() as sess:
        images = []
        for record in tf.python_io.tf_record_iterator(FLAGS.data_file):
            images.append(sess.run(decode, feed_dict={example: record}))
            if len(images) >= size ** 2:
                break
        collage = tf.concat([tf.concat(images[i*size:(i+1)*size], 1) for i in range(size)], 0)
        image = sess.run(tf.image.encode_jpeg(collage))
    with open(os.path.join(os.path.dirname(__file__), 'out.jpg'), 'wb') as f:
        f.write(image)


if __name__ == '__main__':
    tf.app.run()
