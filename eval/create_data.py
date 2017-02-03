import os
import json
import random
import tensorflow as tf

from urllib.parse import urljoin
from urllib.request import urlopen, Request

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('url_base', 'http://localhost:3000',
                           '''API endpoint url''')
tf.app.flags.DEFINE_string('json_file', 'sample.json',
                           '''sampling json file''')
tf.app.flags.DEFINE_integer('num_labels', 120,
                            '''number of target labels''')
tf.app.flags.DEFINE_integer('num_samples', 200,
                            '''number of sample faces''')


class Client:
    def __init__(self, base, email, token):
        self.url_base = base
        self.auth_headers = {
            'X-User-Email': email,
            'X-User-Token': token
        }

    def url(self, path):
        return urljoin(self.url_base, path)

    def get(self, url):
        print('GET {}'.format(url))
        req = Request(url, None, self.auth_headers)
        return json.loads(urlopen(req).read().decode())


def main(argv=None):
    client = Client(
        FLAGS.url_base,
        os.environ['API_AUTH_EMAIL'],
        os.environ['API_AUTH_TOKEN']
    )

    if not os.path.exists(FLAGS.json_file):
        labels = client.get(client.url('labels.json'))
        results = []
        for i in range(FLAGS.num_labels):
            label = labels[i]
            faces = []
            url = client.url('labels/{}/faces.json'.format(label['id']))
            while True:
                result = client.get(url)
                faces += result['faces']
                if result['page']['next']:
                    url = result['page']['next']
                else:
                    break
            results.append(random.sample(faces, FLAGS.num_samples))
        with open(FLAGS.json_file, 'w') as f:
            json.dump(results, f)

    with open(FLAGS.json_file) as f:
        data = json.load(f)
    for index, label in enumerate(data):
        print(index, len(label))


if __name__ == '__main__':
    tf.app.run()
