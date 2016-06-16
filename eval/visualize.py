import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.recognizer import Recognizer
import tensorflow as tf
from eval import inputs

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('images_dir', 'images',
                           """Directory where to write generated images.""")

def main(argv=None):
    r = Recognizer(batch_size=3)
    filepath = os.path.join(FLAGS.data_dir, 'data-00.tfrecords')
    images, labels = inputs([filepath], r.batch_size)
    logits = r.inference(images, FLAGS.num_classes)
    saver = tf.train.Saver(tf.all_variables())

    checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)

        zeros = []
        tensors = []
        targets = [
            {'name': 'pool1:0', 'row':  5, 'col':  8},
            {'name': 'pool2:0', 'row':  5, 'col': 12},
            {'name': 'pool3:0', 'row':  6, 'col': 15},
            {'name': 'pool4:0', 'row':  5, 'col': 12},
        ]
        for target in targets:
            t = sess.graph.get_tensor_by_name(target['name'])
            outputs = tf.split(0, r.batch_size, t)
            for i in range(len(outputs)):
                shape = t.get_shape().as_list()
                maps = [tf.image.resize_image_with_crop_or_pad(1 - x, shape[1] + 2, shape[2] + 2) for x in tf.split(2, shape[3], tf.squeeze(outputs[i], [0]))]
                rows = []
                cols = target['col']
                for row in range(target['row']):
                    rows.append(tf.concat(1, maps[cols * row:cols * (row + 1)]))
                out = tf.concat(0, rows)
                img = tf.image.convert_image_dtype(1 - out, tf.uint8, saturate=True)
                tensors.append(tf.image.encode_png(img, name=t.op.name + '-%02d' % i))

        tf.train.start_queue_runners(sess=sess)
        results = sess.run(tensors)
        for i in range(len(tensors)):
            filename = tensors[i].op.name + '.png'
            print('write %s' % filename)
            with open(os.path.join(os.path.dirname(__file__), '..', FLAGS.images_dir, filename), 'wb') as f:
                f.write(results[i])

if __name__ == '__main__':
    tf.app.run()
