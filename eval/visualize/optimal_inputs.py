import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import tensorflow as tf
from model.recognizer import Recognizer
from eval import inputs

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('target_class', 1,
                           """target class index.""")
tf.app.flags.DEFINE_string('images_dir', 'images',
                           """Directory where to write generated images.""")

def main(argv=None):
    # model definition and train op
    r = Recognizer(batch_size=1)
    # input variable
    with tf.variable_scope('input') as scope:
        v = tf.get_variable('input', shape=(96, 96, 3), initializer=tf.random_uniform_initializer(0.0, 1.0))
    add_noise = v.assign(tf.add(v, tf.random_uniform(v.get_shape(), -0.1, 0.1)))
    # per_image_whitening without relu
    image = tf.mul(tf.clip_by_value(v, 0.0, 1.0), 255.5)
    mean, variance = tf.nn.moments(image, [0, 1, 2])
    pixels = tf.reduce_prod(tf.shape(image))
    stddev = tf.sqrt(tf.maximum(variance, 0))
    image = tf.sub(image, mean)
    image = tf.div(image, tf.maximum(stddev, tf.inv(tf.sqrt(tf.cast(pixels, tf.float32)))))
    # loss and train
    inputs = tf.expand_dims(image, 0)
    logits = r.inference(inputs, FLAGS.num_classes)
    softmax = tf.nn.softmax(logits)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, [FLAGS.target_class])
    train_op = tf.train.AdamOptimizer().minimize(losses, var_list=[v])

    variable_averages = tf.train.ExponentialMovingAverage(r.MOVING_AVERAGE_DECAY)
    variables_to_restore = {}
    for key, value in variable_averages.variables_to_restore().items():
        if not key.startswith('input'):
            variables_to_restore[key] = value
    saver = tf.train.Saver(variables_to_restore)
    checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint)

        for step in range(5000):
            _, loss_value, softmax_value = sess.run([train_op, losses, softmax])
            print('%04d - loss: %f (%f)' % (step, loss_value[0], softmax_value.flatten().tolist()[FLAGS.target_class]))
            if step % 200 == 0:
                sess.run(add_noise)

        # write image to file
        output_image = tf.image.convert_image_dtype(v, tf.uint8, saturate=True)
        filename = 'target-%03d.png' % FLAGS.target_class
        with open(os.path.join(os.path.dirname(__file__), '..', '..', FLAGS.images_dir, filename), 'wb') as f:
            f.write(sess.run(tf.image.encode_png(output_image)))
        filename = 'target-%03d.jpg' % FLAGS.target_class
        with open(os.path.join(os.path.dirname(__file__), '..', '..', FLAGS.images_dir, filename), 'wb') as f:
            f.write(sess.run(tf.image.encode_jpeg(output_image, quality=100, chroma_downsampling=False)))

if __name__ == '__main__':
    tf.app.run()
