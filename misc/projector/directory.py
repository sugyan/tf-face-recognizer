import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import math
from model.recognizer import Recognizer
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Path to model checkpoints.""")
tf.app.flags.DEFINE_string('imgdir', os.path.join(os.path.dirname(__file__), 'images'),
                           """Path to the images directory.""")
tf.app.flags.DEFINE_string('logdir', os.path.join(os.path.dirname(__file__), 'logdir'),
                           """Directory where to write checkpoints.""")


def main(argv=None):
    if not os.path.exists(FLAGS.imgdir):
        raise Exception('%s does not exist' % FLAGS.imgdir)

    r = Recognizer(batch_size=1)
    data = tf.placeholder(tf.string)
    orig_image = tf.image.decode_jpeg(data, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(orig_image, 96, 96)
    image = tf.image.per_image_standardization(image)
    r.inference(tf.expand_dims(image, axis=0), 1)
    fc5 = tf.get_default_graph().get_tensor_by_name('fc5/fc5:0')
    fc6 = tf.get_default_graph().get_tensor_by_name('fc6/fc6:0')
    with tf.Session() as sess:
        variable_averages = tf.train.ExponentialMovingAverage(r.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        for name, v in variables_to_restore.items():
            try:
                tf.train.Saver([v]).restore(sess, FLAGS.checkpoint_path)
            except Exception:
                print('initialize %s' % name)
                sess.run(tf.variables_initializer([v]))

        outputs = {
            'fc5': [],
            'fc6': [],
            'images': []
        }
        for file in os.listdir(FLAGS.imgdir):
            if not file.endswith('.jpg'):
                continue
            print('processing {}...'.format(file))
            with open(os.path.join(FLAGS.imgdir, file), 'rb') as f:
                results = sess.run({
                    'fc5': fc5,
                    'fc6': fc6,
                    'image': orig_image
                }, feed_dict={data: f.read()})
            outputs['fc5'].append(results['fc5'].flatten().tolist())
            outputs['fc6'].append(results['fc6'].flatten().tolist())
            outputs['images'].append(results['image'])

        # write to sprite image file
        image_path = os.path.join(FLAGS.logdir, 'sprite.jpg')
        images = outputs['images']
        rows = []
        size = int(math.sqrt(len(images))) + 1
        while len(images) < size * size:
            images.append(np.zeros((112, 112, 3), dtype=np.uint8))
        for i in range(size):
            rows.append(tf.concat(1, images[i*size:(i+1)*size]))
        jpeg = tf.image.encode_jpeg(tf.concat(0, rows))
        with open(image_path, 'wb') as f:
            f.write(sess.run(jpeg))
        # add embeding data
        targets = [
            tf.Variable(np.stack(outputs['fc5']), name='fc5'),
            tf.Variable(np.stack(outputs['fc6']), name='fc6'),
        ]
        config = projector.ProjectorConfig()
        for v in targets:
            embedding = config.embeddings.add()
            embedding.tensor_name = v.name
            # embedding.metadata_path = metadata_path
            embedding.sprite.image_path = image_path
            embedding.sprite.single_image_dim.extend([112, 112])
        sess.run(tf.variables_initializer(targets))
        summary_writer = tf.summary.FileWriter(FLAGS.logdir)
        projector.visualize_embeddings(summary_writer, config)
        graph_saver = tf.train.Saver(targets)
        graph_saver.save(sess, os.path.join(FLAGS.logdir, 'model.ckpt'))


if __name__ == '__main__':
    tf.app.run()
