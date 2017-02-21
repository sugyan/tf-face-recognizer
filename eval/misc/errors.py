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
    image = tf.image.per_image_standardization(image)
    images = tf.expand_dims(image, axis=0)
    labels = tf.expand_dims(features['label'], axis=0)

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import model
    logits = tf.nn.softmax(model.inference(images, FLAGS.num_classes))
    _, indices = tf.nn.top_k(logits)
    top = indices[0]
    # correct = tf.nn.in_top_k(logits, labels, 1)
    # variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)

    with tf.Session() as sess:
        tf.train.Saver(tf.trainable_variables()).restore(sess, FLAGS.checkpoint_path)
        # tf.train.Saver(variable_averages.variables_to_restore()).restore(sess, FLAGS.checkpoint_path)

        ok, ng = 0, 0
        error_top = {}
        error_ans = {}
        for i, record in enumerate(tf.python_io.tf_record_iterator(FLAGS.data_file)):
            # result = sess.run(correct, feed_dict={example: record})
            top_value, labels_value = sess.run([top, labels], feed_dict={example: record})
            # if result[0]:
            if top_value[0] == labels_value[0]:
                ok += 1
            else:
                # print('{:04d}: {} ({}, {})'.format(i, result[0], top_k_value, labels_value))
                print('{:04d}: {:3d} - {:3d}'.format(i, top_value[0], labels_value[0]))
                if top_value[0] not in error_top:
                    error_top[top_value[0]] = 0
                error_top[top_value[0]] += 1
                if labels_value[0] not in error_ans:
                    error_ans[labels_value[0]] = 0
                error_ans[labels_value[0]] += 1
                ng += 1
    print('{}/{} ({:.3f} %)'.format(ok, ok + ng, 100.0 * ok / (ok + ng)))
    print('top:')
    for k, v in sorted(error_top.items(), key=lambda x: x[1], reverse=True):
        print('{:3d}: {:3d}'.format(k, v))
    print('ans:')
    for k, v in sorted(error_ans.items(), key=lambda x: x[1], reverse=True):
        print('{:3d}: {:3d}'.format(k, v))


if __name__ == '__main__':
    tf.app.run()
