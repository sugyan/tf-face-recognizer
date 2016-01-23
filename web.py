from models import v2

from flask import Flask, jsonify, request
import tensorflow as tf

import base64
import urllib
import os
import gzip

v2.BATCH_SIZE = 1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Directory where to read model checkpoints.""")

images = tf.placeholder(tf.float32, shape=(1, v2.INPUT_SIZE, v2.INPUT_SIZE, 3))
logits = tf.nn.softmax(v2.inference(images))

sess = tf.Session()
variable_averages = tf.train.ExponentialMovingAverage(v2.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
if not os.path.isfile(FLAGS.checkpoint_path):
    print 'No checkpoint file found'
    urllib.urlretrieve(os.environ['DOWNLOAD_URL'], FLAGS.checkpoint_path + '.gz')
    open(FLAGS.checkpoint_path, 'wb').write(gzip.open(FLAGS.checkpoint_path + '.gz').read())
saver.restore(sess, FLAGS.checkpoint_path)

app = Flask(__name__)
app.debug = True

@app.route('/', methods=['POST'])
def api():
    results = []
    for image in request.form.getlist('images'):
        data = base64.b64decode(image.split(',')[1])
        if image.startswith('data:image/jpeg;base64,'):
            decoded = tf.image.decode_jpeg(data, channels=3)
        if image.startswith('data:image/png;base64,'):
            decoded = tf.image.decode_png(data, channels=3)
        decoded.set_shape(sess.run(decoded).shape)
        resized = tf.image.resize_image_with_crop_or_pad(decoded, v2.INPUT_SIZE, v2.INPUT_SIZE)
        inputs = tf.image.per_image_whitening(resized)
        inputs = sess.run(tf.expand_dims(inputs, 0))
        output = sess.run(logits, feed_dict={images: inputs})
        results.append(output.flatten().tolist())
    return jsonify(results=results)
