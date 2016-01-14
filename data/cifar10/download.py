import urllib
import random
import sys
import os

url_base = sys.argv[1]
size = 32

targets = (
    {'label_id': 1, 'num': 200},
    {'label_id': 2, 'num': 200},
    {'label_id': 3, 'num': 200},
    {'label_id': 4, 'num': 200},
    {'label_id': 5, 'num': 200},
    {'label_id': 6, 'num': 128},
)
results = []
for target in targets:
    target['size'] = size
    params = urllib.urlencode(target)
    data = urllib.urlopen(url_base + '/faces/cifar10?%s' % params).read()
    chunk_size = size ** 2 * 3 + 1
    results.extend([data[i:i+chunk_size] for i in range(0, len(data), chunk_size)])
print '%d results.' % len(results)
random.shuffle(results)

trains = [results[i * 200:(i + 1) * 200] for i in range(5)]
test = results[200 * 5:200 * 5 + 128]
for i in range(5):
    f = open(os.path.join(os.path.dirname(__file__), 'cifar-10-batches-bin', 'data_batch_%d.bin' % (i + 1)), 'w')
    f.write(''.join(trains[i]))
    f.close()
with open(os.path.join(os.path.dirname(__file__), 'cifar-10-batches-bin', 'test_batch.bin'), 'w') as f:
    f.write(''.join(test))
