import os
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('distortion', 0,
                            """Distortion mode.""")


def main(argv=None):
    with open(os.path.join(os.path.dirname(__file__), 'face.png'), 'rb') as f:
        png = f.read()
    image = tf.image.decode_png(png, channels=3)

    if FLAGS.distortion == 0:
        image = tf.to_float(tf.random_crop(image, [96, 96, 3]))
    else:
        bounding_boxes = tf.div(tf.constant([[[8, 8, 104, 104]]], dtype=tf.float32), 112.0)
        begin, size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image), bounding_boxes,
            min_object_covered=(80.0*80.0)/(96.0*96.0),
            aspect_ratio_range=[9.0/10.0, 10.0/9.0])
        image = tf.slice(image, begin, size)
        image = tf.image.resize_images(image, [96, 96])
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_hue(image, max_delta=0.04)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

    images = tf.unstack(tf.train.batch([image], 64))
    montage = tf.concat([tf.concat(images[x*8:(x+1)*8], 1) for x in range(8)], 0)
    montage = tf.image.encode_jpeg(tf.image.convert_image_dtype(tf.div(montage, 255.0), tf.uint8, saturate=True))

    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)

        with open(os.path.join(os.path.dirname(__file__), 'out.jpg'), 'wb') as f:
            f.write(sess.run(montage))


if __name__ == '__main__':
    tf.app.run()
