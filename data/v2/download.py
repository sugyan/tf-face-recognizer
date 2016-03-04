import tensorflow as tf
import logging
import urllib
import random
import sys
import os
import json

url_base = sys.argv[1]

# config
targets, labels = [], {}
samples = 0
url = url_base + '.json'
while True:
    results = json.loads(urllib.urlopen(url).read())
    indexed = False
    for label in results['labels']:
        index_number = label['index_number']
        if index_number is not None:
            indexed = True
            targets.append({
                'index': index_number,
                'sample': label['faces_count']
            })
            samples += label['faces_count']
            labels[index_number] = label
    url = results['page']['next']
    if not indexed:
        break
targets.append({
    'index': 0,
    'sample': samples
})

# labels data
with open(os.path.join(os.path.dirname(__file__), 'tfrecords', 'labels.json'), 'w') as f:
    f.write(json.dumps(labels))
# download source data
for target in targets:
    params = urllib.urlencode({ 'sample': target['sample'] })
    url = url_base + '/faces/tfrecords/%d?%s' % (target['index'], params)
    filename = os.path.join(os.path.dirname(__file__), 'tfrecords', '%03d.tfrecords' % target['index'])
    print urllib.urlretrieve(url, filename)

print samples * 2
