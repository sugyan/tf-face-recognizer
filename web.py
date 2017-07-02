from flask import Flask, jsonify, request
import tensorflow as tf

import base64
import urllib.request
import os
import json

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_path', '/tmp/model.pb',
                           """Path to model data.""")
tf.app.flags.DEFINE_integer('port', 5000,
                            """Application port.""")
tf.app.flags.DEFINE_integer('top_k', 5,
                            """Finds the k largest entries""")
tf.app.flags.DEFINE_integer('input_size', 96,
                            """Size of input image""")

sess = tf.Session()

# load model data, get top_k
if not os.path.isfile(FLAGS.model_path):
    print('No model data file found')
    urllib.request.urlretrieve(os.environ['MODEL_DOWNLOAD_URL'], FLAGS.model_path)
graph_def = tf.GraphDef()
with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name='')
fc7 = sess.graph.get_tensor_by_name('fc7/fc7:0')
top_values, top_indices = tf.nn.top_k(tf.nn.softmax(fc7), k=FLAGS.top_k)
# retrieve labels
labels = json.loads(sess.run(sess.graph.get_tensor_by_name('labels:0')).decode())
print('{} labels loaded.'.format(len(labels)))

# Flask setup
app = Flask(__name__)
app.debug = True


@app.route('/labels')
def label():
    return jsonify(labels=labels)


@app.route('/', methods=['POST'])
def api():
    results = []
    ops = [top_values, top_indices]
    for image in request.form.getlist('images'):
        values, indices = sess.run(ops, feed_dict={'contents:0': base64.b64decode(image.split(',')[1])})
        top_k = []
        for i in range(FLAGS.top_k):
            top_k.append({
                'label': labels.get(str(indices.flatten().tolist()[i]), {}),
                'value': values.flatten().tolist()[i],
            })
        results.append({'top': top_k})
    return jsonify(results=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLAGS.port)
