import tensorflow as tf
import logging
import urllib
import random
import sys
import os
import json

url_base = sys.argv[1]

# remove old records
data_dir = os.path.join(os.path.dirname(__file__), 'tfrecords')
for f in os.listdir(data_dir):
    if f.endswith('.tfrecords'):
        os.remove(os.path.join(data_dir, f))

# config
targets, labels = [], {}
url = url_base + '/labels.json'
samples = 0
for label in json.loads(urllib.urlopen(url).read()):
    index_number = label['index_number']
    if index_number is not None:
        samples += label['faces_count']
        targets.append({
            'index': index_number,
            'sample': label['faces_count']
        })
        labels[index_number] = label
targets.append({
    'index': 0,
    'sample': samples / 2
})

# labels data
with open(os.path.join(os.path.dirname(__file__), 'tfrecords', 'labels.json'), 'w') as f:
    f.write(json.dumps(labels))
# download source data
for target in targets:
    url = url_base + '/faces/tfrecords/%d?%s' % (target['index'], urllib.urlencode({ 'sample': target['sample'] }))
    filename = os.path.join(data_dir, '%03d.tfrecords' % target['index'])
    print urllib.urlretrieve(url, filename)

print samples + samples / 2
