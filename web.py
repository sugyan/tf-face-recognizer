from models import v2

from flask import Flask, jsonify, request
import tensorflow as tf

import base64
import urllib
import os
import json

v2.BATCH_SIZE = 1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('port', 5000,
                           """Application port.""")
tf.app.flags.DEFINE_integer('top_k', 5,
                           """Finds the k largest entries""")

# Flask setup
app = Flask(__name__)
app.debug = True

# Logits setup
input_data = tf.placeholder(tf.string)
decoded = tf.image.decode_jpeg(input_data)
resized = tf.image.resize_images(decoded, v2.INPUT_SIZE, v2.INPUT_SIZE)
inputs = tf.expand_dims(tf.image.per_image_whitening(resized), 0)
logits = v2.inference(inputs)
outputs = tf.nn.softmax(logits)
top_results = tf.nn.top_k(outputs, k=FLAGS.top_k)

global_step = tf.Variable(0, name='global_step', trainable=False)
labels = tf.Variable(tf.bytes(), name='labels', trainable=False)

sess = tf.Session()
variable_averages = tf.train.ExponentialMovingAverage(v2.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

if os.path.isfile(FLAGS.checkpoint_path):
    saver.restore(sess, FLAGS.checkpoint_path)
else:
    print 'No checkpoint file found'
    urllib.urlretrieve(os.environ['CHECKPOINT_DOWNLOAD_URL'], FLAGS.checkpoint_path)
    saver.restore(sess, FLAGS.checkpoint_path)
labels = json.loads(sess.run(labels))

@app.route('/', methods=['POST'])
def api():
    results = []
    for image in request.form.getlist('images'):
        values, indices = sess.run(top_results, feed_dict={input_data:base64.b64decode(image.split(',')[1])})
        result = []
        for i in range(FLAGS.top_k):
            result.append({
                'label': labels.get(str(indices.flatten().tolist()[i]), {}),
                'value': values.flatten().tolist()[i],
            })
        results.append(result)
    return jsonify(results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLAGS.port)
