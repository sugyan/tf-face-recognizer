import tensorflow as tf
import logging
import urllib
import random
import sys
import os
import json

url_base = sys.argv[1]

# config
targets = (
    {'label_id': 1, 'num': 220},
    {'label_id': 2, 'num': 220},
    {'label_id': 3, 'num': 220},
    {'label_id': 4, 'num': 220},
    {'label_id': 5, 'num': 220},
)

# download source data
sources = []
for target in targets:
    params = urllib.urlencode({ 'num': target['num'] })
    data = json.loads(urllib.urlopen(url_base + '/labels/%s/sample.json?%s' % (target['label_id'], params)).read())
    for datum in data['faces']:
        sources.append({
            'label': data['label_index'],
            'id': datum['id'],
            'image_url': datum['image_url'],
            'image_size': datum['image_size'],
        })
random.shuffle(sources)

# fetch all data and write tfrecords
destinations = (
    { 'name': 'data1.tfrecords', 'num': 200 },
    { 'name': 'data2.tfrecords', 'num': 200 },
    { 'name': 'data3.tfrecords', 'num': 200 },
    { 'name': 'data4.tfrecords', 'num': 200 },
    { 'name': 'data5.tfrecords', 'num': 200 },
    { 'name': 'test.tfrecords',  'num': 100 },
)
for destination in destinations:
    filename = os.path.join(os.path.dirname(__file__), 'tfrecords', destination['name'])
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(destination['num']):
        source = sources.pop(0)
        data = urllib.urlopen(source['image_url']).read()
        assert source['image_size'] == len(data)
        print '%s: write face id:%d (%d bytes) with label [%d].' % (
            destination['name'], source['id'], len(data), source['label'],
        )
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[source['label']])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
        }))
        writer.write(example.SerializeToString())
