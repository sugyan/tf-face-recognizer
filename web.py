from models import v2

from flask import Flask, jsonify, request
from flask.ext.sqlalchemy import SQLAlchemy
import tensorflow as tf

import base64
import urllib
import os

v2.BATCH_SIZE = 1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('port', 5000,
                           """Application port.""")

# Flask setup
app = Flask(__name__)
app.debug = True
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)
db.create_all()

class CheckPoint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.LargeBinary)

    def __init__(self, id):
        self.id = id

# Logits setup
images = tf.placeholder(tf.float32, shape=(1, v2.INPUT_SIZE, v2.INPUT_SIZE, 3))
logits = tf.nn.softmax(v2.inference(images))

sess = tf.Session()
variable_averages = tf.train.ExponentialMovingAverage(v2.MOVING_AVERAGE_DECAY)
variables_to_restore = {}
for v in tf.all_variables():
    if v in tf.trainable_variables():
        restore_name = variable_averages.average_name(v)
    else:
        restore_name = v.op.name
    variables_to_restore[restore_name] = v
saver = tf.train.Saver(variables_to_restore)


if os.path.isfile(FLAGS.checkpoint_path):
    saver.restore(sess, FLAGS.checkpoint_path)
else:
    print 'No checkpoint file found'
    checkpoint = CheckPoint.query.get(1)
    if checkpoint:
        open(FLAGS.checkpoint_path, 'wb').write(checkpoint.data)
        saver.restore(sess, FLAGS.checkpoint_path)
        del checkpoint

@app.route('/', methods=['POST'])
def api():
    results = []
    for image in request.form.getlist('images'):
        data = base64.b64decode(image.split(',')[1])
        if image.startswith('data:image/jpeg;base64,'):
            decoded = tf.image.decode_jpeg(data, channels=3)
        if image.startswith('data:image/png;base64,'):
            decoded = tf.image.decode_png(data, channels=3)
        decoded.set_shape(decoded.eval(session=tf.Session()).shape)
        resized = tf.image.resize_image_with_crop_or_pad(decoded, v2.INPUT_SIZE, v2.INPUT_SIZE)
        inputs = tf.image.per_image_whitening(resized)
        inputs = tf.expand_dims(inputs, 0).eval(session=tf.Session())
        output = sess.run(logits, feed_dict={images: inputs})
        results.append(output.flatten().tolist())
    return jsonify(results=results)

@app.route('/checkpoint', methods=['POST'])
def checkpoint():
    checkpoint = CheckPoint.query.get(1)
    if not checkpoint:
        checkpoint = CheckPoint(1)
        db.session.add(checkpoint)
        db.session.commit()
        checkpoint = CheckPoint.query.get(1)
    checkpoint.data = request.files['file'].read()
    db.session.commit()
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLAGS.port)
