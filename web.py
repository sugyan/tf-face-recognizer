from model.recognizer import Recognizer

from flask import Flask, jsonify, request
import tensorflow as tf

import base64
import urllib.request
import os
import json

r = Recognizer(batch_size=1)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('port', 5000,
                           """Application port.""")
tf.app.flags.DEFINE_integer('top_k', 5,
                           """Finds the k largest entries""")

if not os.path.isfile(FLAGS.checkpoint_path):
    print('No checkpoint file found')
    urllib.request.urlretrieve(os.environ['CHECKPOINT_DOWNLOAD_URL'], FLAGS.checkpoint_path)

# Flask setup
app = Flask(__name__)
app.debug = True

# Logits setup
sess = tf.Session()

# restore label data
labels = tf.Variable(tf.bytes(), name='labels', trainable=False)
labels_saver = tf.train.Saver([labels])
labels_saver.restore(sess, FLAGS.checkpoint_path)
labels = json.loads(sess.run(labels).decode())
print('%d labels' % len(labels))

input_data = tf.placeholder(tf.string)
decoded = tf.image.decode_jpeg(input_data, channels=3)
resized = tf.image.resize_images(decoded, r.INPUT_SIZE, r.INPUT_SIZE)
inputs = tf.expand_dims(tf.image.per_image_whitening(resized), 0)
logits = r.inference(inputs, len(labels.keys()) + 1)
fc6 = tf.get_default_graph().get_tensor_by_name('fc6/fc6:0')
top_values, top_indices = tf.nn.top_k(tf.nn.softmax(logits), k=FLAGS.top_k)

# restore model variables
variable_averages = tf.train.ExponentialMovingAverage(r.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, FLAGS.checkpoint_path)

@app.route('/', methods=['POST'])
def api():
    results = []
    ops = [top_values, top_indices]
    if 'fc6' in request.form:
        ops.append(fc6)
    for image in request.form.getlist('images'):
        outputs = sess.run(ops, feed_dict={input_data:base64.b64decode(image.split(',')[1])})
        values, indices = outputs[0:2]
        result = {
            'top': []
        }
        if len(outputs) > 2:
            result['fc6'] = outputs[2].flatten().tolist()
        for i in range(FLAGS.top_k):
            result['top'].append({
                'label': labels.get(str(indices.flatten().tolist()[i]), {}),
                'value': values.flatten().tolist()[i],
            })
        results.append(result)
    return jsonify(results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLAGS.port)
