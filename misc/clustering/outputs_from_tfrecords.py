import os
import sys
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_path', '/tmp/model.pb',
                           """Directory where to read model data.""")
tf.app.flags.DEFINE_integer('input_size', 96,
                            """Size of input image""")


def main(argv=None):
    if not os.path.isfile(FLAGS.model_path):
        print('No model data file found: {}'.format(FLAGS.model_path))
        sys.exit()
    # load graph
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    graph = tf.get_default_graph()
    fc5 = graph.get_tensor_by_name('fc5/fc5:0')
    fc6 = graph.get_tensor_by_name('fc6/fc6:0')
    print(fc5, fc6)

    # r = Recognizer(batch_size=1)
    # data = tf.placeholder(tf.string)
    # image = tf.image.decode_jpeg(data, channels=3)
    # image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.input_size, FLAGS.input_size)
    # image = tf.image.per_image_standardization(image)
    # inputs = tf.expand_dims(image, axis=0)
    # r.inference(inputs, 0)
    # fc5 = tf.get_default_graph().get_tensor_by_name('fc5/fc5:0')
    # fc6 = tf.get_default_graph().get_tensor_by_name('fc6/fc6:0')

    # with tf.Session() as sess:
    #     variable_averages = tf.train.ExponentialMovingAverage(r.MOVING_AVERAGE_DECAY)
    #     variables_to_restore = variable_averages.variables_to_restore()
    #     for name, v in variables_to_restore.items():
    #         try:
    #             tf.train.Saver([v]).restore(sess, FLAGS.checkpoint_path)
    #         except Exception:
    #             print('initialize %s' % name)
    #             sess.run(tf.variables_initializer([v]))

    #     outputs = {}
    #     dirname = os.path.join(os.path.dirname(__file__), 'images')
    #     for filename in os.listdir(dirname):
    #         if not filename.endswith('.jpg'):
    #             continue
    #         with open(os.path.join(dirname, filename), 'rb') as f:
    #             results = sess.run({'fc5': fc5, 'fc6': fc6}, feed_dict={data: f.read()})
    #         outputs[filename] = {
    #             'fc5': results['fc5'].flatten().tolist(),
    #             'fc6': results['fc6'].flatten().tolist(),
    #         }
    # for out in ['fc5', 'fc6']:
    #     filename = os.path.join(os.path.dirname(__file__), '%s.csv' % out)
    #     with open(filename, 'w') as f:
    #         for name, values in outputs.items():
    #             f.write(','.join([name] + [str(x) for x in values[out]]) + '\n')


if __name__ == '__main__':
    tf.app.run()
