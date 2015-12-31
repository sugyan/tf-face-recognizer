from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf
import json

tf.app.flags.DEFINE_string('image_file', 'image', '''image file name''')

cifar10.IMAGE_SIZE = 48
cifar10.NUM_CLASSES = 6

def recognize(filename):
    with open(filename) as f:
        data = tf.image.decode_jpeg(f.read())
        with tf.Session() as sess:
            img = sess.run(data)
    shape = [1]
    shape.extend(img.shape)
    reshaped = tf.reshape(img, shape)
    resized = tf.image.resize_images(reshaped, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE)
    logits = cifar10.inference(resized)
    saver = tf.train.Saver(tf.all_variables())

    checkpoint_dir = 'train'
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.initialize_all_variables())
        return sess.run(logits).flatten().tolist()

def main(argv):
    result = recognize(tf.app.flags.FLAGS.image_file)
    print json.dumps(result)

if __name__ == '__main__':
    tf.app.run()
