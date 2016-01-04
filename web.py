from flask import Flask, jsonify, request
from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf
import json

from gunicorn.app.base import BaseApplication
from gunicorn.six import iteritems

cifar10.NUM_CLASSES = 6

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', 'train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('image_size', 32,
                           """Image size.""")

images = tf.placeholder(tf.float32, shape=(1, FLAGS.image_size, FLAGS.image_size, 3))
logits = tf.nn.softmax(cifar10.inference(images))

sess = tf.Session()
saver = tf.train.Saver(tf.all_variables())
ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print('No checkpoint file found')

app = Flask(__name__)
app.debug = True

@app.route('/recognize', methods=['POST'])
def api():
    f = request.files['image']
    data = f.read()
    for decoder in (tf.image.decode_jpeg, tf.image.decode_png):
        try:
            decoded = decoder(data, channels=3)
            if decoded.eval(session=tf.Session()) is not None:
                break
        except tf.errors.InvalidArgumentError as e:
            app.logger.warning(e.message)
    inputs = tf.reshape(decoded, decoded.eval(session=tf.Session()).shape)
    inputs = tf.image.per_image_whitening(inputs)
    inputs = tf.image.resize_images(tf.expand_dims(inputs, 0), FLAGS.image_size, FLAGS.image_size)
    output = sess.run(logits, feed_dict={images: inputs.eval(session=tf.Session())})
    return jsonify(result=output.flatten().tolist())

@app.route('/')
def root():
    return 'OK'

if __name__ == '__main__':
    app.run()
