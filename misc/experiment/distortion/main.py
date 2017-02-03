import os
import tensorflow as tf


def main(argv=None):
    with open(os.path.join(os.path.dirname(__file__), 'face.png'), 'rb') as f:
        png = f.read()
    image = tf.image.decode_png(png, channels=3)

    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        [[[8.0/112.0, 8.0/112.0, (112.0-8.0)/112.0, (112.0-8.0)/112.0]]],
        min_object_covered=0.9)
    image = tf.slice(image, begin, size)
    resize = tf.random_uniform([2], minval=48, maxval=144, dtype=tf.int32)
    image = tf.image.resize_images(image, resize, method=2)
    image = tf.image.resize_images(image, [96, 96], method=2)
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_hue(image, max_delta=0.04)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

    images = tf.train.batch([tf.image.per_image_standardization(image)], 20)
    summary = tf.summary.image('images', images, max_outputs=20)
    writer = tf.summary.FileWriter(os.path.dirname(__file__))

    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)
        summary_value, begin_value, size_value, resize_value = sess.run([summary, begin, size, resize])
        print(begin_value, size_value, resize_value)
        writer.add_summary(summary_value)


if __name__ == '__main__':
    tf.app.run()
