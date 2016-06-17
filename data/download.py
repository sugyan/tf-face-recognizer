import logging
import random
import sys
import os
import json
import codecs

try:
    from urllib.parse import urlencode
    from urllib.request import urlopen
except ImportError:
    from urllib import urlencode
    from urllib2 import urlopen


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

reader = codecs.getreader('utf-8')
for label in json.load(reader(urlopen(url))):
    index_number = label['index_number']
    if index_number is not None:
        sample = 120 if index_number > 0 and label['faces_count'] > 120 else label['faces_count']
        samples += sample
        targets.append({
            'index': index_number,
            'sample': sample
        })
        labels[index_number] = label
targets.append({
    'index': 0,
    'sample': int(samples / 3)
})

# labels data
with open(os.path.join(os.path.dirname(__file__), 'tfrecords', 'labels.json'), 'w') as f:
    f.write(json.dumps(labels))
# download source data
for target in targets:
    url = url_base + '/faces/tfrecords/%d?%s' % (target['index'], urlencode({ 'sample': target['sample'] }))
    number = 0 if target['index'] == 0 else (target['index'] - 1) / 10 + 1
    filename = os.path.join(data_dir, '%02d.tfrecords' % number)
    if target['sample'] <= 60:
        with open(filename, 'ab') as f:
            f.write(urlopen(url).read())
            f.write(urlopen(url).read())
        print('%s (%d: %d x2)' % (filename, target['index'], target['sample']))
    else:
        with open(filename, 'ab') as f:
            f.write(urlopen(url).read())
        print('%s (%d: %d)' % (filename, target['index'], target['sample']))

print(samples + int(samples / 3))
