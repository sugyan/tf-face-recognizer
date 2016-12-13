import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.recognizer import Recognizer
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Path to model checkpoints.""")
tf.app.flags.DEFINE_string('input_file', 'data.tfrecords',
                           """Path to the TFRecord data.""")
tf.app.flags.DEFINE_string('logdir', os.path.join(os.path.dirname(__file__), 'logdir'),
                           """Directory where to write checkpoints.""")


def inputs(files, batch_size=0):
    fqueue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    key, value = reader.read(fqueue)
    features = tf.parse_single_example(value, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })
    label = features['label']
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 96, 96)
    return tf.train.batch(
        [tf.image.per_image_standardization(image), image, label], batch_size
    )


def main(argv=None):
    filepath = FLAGS.input_file
    if not os.path.exists(filepath):
        raise Exception('%s does not exist' % filepath)

    r = Recognizer(batch_size=900)
    input_images, orig_images, labels = inputs([filepath], batch_size=r.batch_size)
    r.inference(input_images, 1)
    fc5 = tf.get_default_graph().get_tensor_by_name('fc5/fc5:0')
    fc6 = tf.get_default_graph().get_tensor_by_name('fc6/fc6:0')
    with tf.Session() as sess:
        variable_averages = tf.train.ExponentialMovingAverage(r.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        for name, v in variables_to_restore.items():
            try:
                tf.train.Saver([v]).restore(sess, FLAGS.checkpoint_path)
            except Exception:
                sess.run(tf.variables_initializer([v]))

        tf.train.start_queue_runners(sess=sess)
        outputs = sess.run({'fc5': fc5, 'fc6': fc6, 'images': orig_images, 'labels': labels})

        # write to metadata file
        metadata_path = os.path.join(FLAGS.logdir, 'metadata.tsv')
        with open(metadata_path, 'w') as f:
            for index in outputs['labels']:
                f.write('%d\n' % index)
        # write to sprite image file
        image_path = os.path.join(FLAGS.logdir, 'sprite.jpg')
        unpacked = tf.unpack(outputs['images'], 900)
        rows = []
        for i in range(30):
            rows.append(tf.concat(1, unpacked[i*30:(i+1)*30]))
        jpeg = tf.image.encode_jpeg(tf.concat(0, rows))
        with open(image_path, 'wb') as f:
            f.write(sess.run(jpeg))
        # add embedding data
        targets = [tf.Variable(e, name=name) for name, e in outputs.items() if name.startswith('fc')]
        config = projector.ProjectorConfig()
        for v in targets:
            embedding = config.embeddings.add()
            embedding.tensor_name = v.name
            embedding.metadata_path = metadata_path
            embedding.sprite.image_path = image_path
            embedding.sprite.single_image_dim.extend([96, 96])
        sess.run(tf.variables_initializer(targets))
        summary_writer = tf.train.SummaryWriter(FLAGS.logdir)
        projector.visualize_embeddings(summary_writer, config)
        graph_saver = tf.train.Saver(targets)
        graph_saver.save(sess, os.path.join(FLAGS.logdir, 'model.ckpt'))


if __name__ == '__main__':
    tf.app.run()
