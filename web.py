from flask import Flask, jsonify, request
from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf
import base64
import urllib
import os

cifar10.NUM_CLASSES = 6

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('download_url', 'http://',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('image_size', 32,
                           """Image size.""")
tf.app.flags.DEFINE_integer('port', 5000,
                           """Application port.""")

images = tf.placeholder(tf.float32, shape=(1, FLAGS.image_size, FLAGS.image_size, 3))
logits = tf.nn.softmax(cifar10.inference(images))

sess = tf.Session()
saver = tf.train.Saver(tf.all_variables())
if not os.path.isfile(FLAGS.checkpoint_path):
    print 'No checkpoint file found'
    print urllib.urlretrieve(FLAGS.download_url, FLAGS.checkpoint_path)
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
        inputs = tf.reshape(decoded, decoded.eval(session=tf.Session()).shape)
        inputs = tf.image.per_image_whitening(inputs)
        inputs = tf.image.resize_images(tf.expand_dims(inputs, 0), FLAGS.image_size, FLAGS.image_size)
        output = sess.run(logits, feed_dict={images: inputs.eval(session=tf.Session())})
        results.append(output.flatten().tolist())
    return jsonify(results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLAGS.port)
