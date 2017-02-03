import os
import sys
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_path', '/tmp/model.pb',
                           """Directory where to read model data.""")
tf.app.flags.DEFINE_string('file', 'data.tfrecords',
                           """Path to the tfrecord file.""")


def main(argv=None):
    if not os.path.isfile(FLAGS.model_path):
        print('No model data file found: {}'.format(FLAGS.model_path))
        sys.exit()
    # load graph
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    # fc5, fc6
    graph = tf.get_default_graph()
    fc5 = graph.get_tensor_by_name('fc5/fc5:0')
    fc6 = graph.get_tensor_by_name('fc6/fc6:0')

    example = tf.placeholder(tf.string)
    features = tf.parse_single_example(example, features={
        'id': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })
    with tf.Session() as sess:
        # read records and run for getting outputs
        outputs = {}
        for i, record in enumerate(tf.python_io.tf_record_iterator(FLAGS.file)):
            print('processing {:04d}...'.format(i))
            data = sess.run(features, feed_dict={example: record})
            results = sess.run({'fc5': fc5, 'fc6': fc6}, feed_dict={'contents:0': data['image_raw']})
            outputs[data['id']] = {
                'fc5': results['fc5'].flatten().tolist(),
                'fc6': results['fc6'].flatten().tolist(),
            }
    # write outputs to CSV
    for out in ['fc5', 'fc6']:
        filename = os.path.join(os.path.dirname(__file__), '{}.csv'.format(out))
        with open(filename, 'w') as f:
            for name, values in outputs.items():
                f.write(','.join(['{:07d}'.format(name)] + [str(x) for x in values[out]]) + '\n')


if __name__ == '__main__':
    tf.app.run()
