import os
import sys
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_file',
                           os.path.join(os.path.dirname(__file__), '..', 'data', 'tfrecords', 'data-00.tfrecords'),
                           """Path to the TFRecord.""")
tf.app.flags.DEFINE_string('checkpoint_path',
                           os.path.join(os.path.dirname(__file__), '..', 'logdir', 'model.ckpt-80000'),
                           """Path to read model checkpoint.""")
tf.app.flags.DEFINE_integer('num_classes', 120,
                            """Number of classes.""")


def main(argv=None):
    example = tf.placeholder(tf.string)
    features = tf.parse_single_example(example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })
    decode = tf.image.decode_jpeg(features['image_raw'], channels=3)
    image = tf.image.resize_image_with_crop_or_pad(decode, 96, 96)
    inputs = tf.expand_dims(tf.image.per_image_standardization(image), axis=0)
    labels = tf.expand_dims(features['label'], axis=0)

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import model
    logits = tf.nn.softmax(model.inference(inputs, FLAGS.num_classes))
    _, indices = tf.nn.top_k(logits)
    top = indices[0]
    # variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)

    with tf.Session() as sess:
        tf.train.Saver(tf.trainable_variables()).restore(sess, FLAGS.checkpoint_path)
        # tf.train.Saver(variable_averages.variables_to_restore()).restore(sess, FLAGS.checkpoint_path)

        ok, ng = 0, 0
        errors = {}
        error_images = []
        for i, record in enumerate(tf.python_io.tf_record_iterator(FLAGS.data_file)):
            top_value, labels_value, image_value = sess.run([top, labels, image], feed_dict={example: record})
            if top_value[0] == labels_value[0]:
                ok += 1
            else:
                # print('{:04d}: {} ({}, {})'.format(i, result[0], top_k_value, labels_value))
                print('{:04d}: {:3d} - {:3d}'.format(i, top_value[0], labels_value[0]))
                if labels_value[0] not in errors:
                    errors[labels_value[0]] = 0
                errors[labels_value[0]] += 1
                ng += 1
                if len(error_images) < 100:
                    error_images.append(image_value)
        size = 10
        collage = tf.concat([tf.concat(error_images[i*size:(i+1)*size], 1) for i in range(size)], 0)
        with open(os.path.join(os.path.dirname(__file__), 'errors.jpg'), 'wb') as f:
            f.write(sess.run(tf.image.encode_jpeg(collage)))
    print('{}/{} ({:.3f} %)'.format(ok, ok + ng, 100.0 * ok / (ok + ng)))
    print('errors:')
    for k, v in sorted(errors.items(), key=lambda x: x[1], reverse=True):
        if v >= 5:
            print('{:3d}: {:3d}'.format(k, v))


if __name__ == '__main__':
    tf.app.run()
