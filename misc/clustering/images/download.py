import os
import json
from urllib.request import urlopen, urlretrieve

url = os.environ['API_ENDPOINT'] + '/faces/random.json'
for i in range(100):
    data = json.loads(urlopen(url).read().decode())
    filename, _ = urlretrieve(data['image_url'], os.path.join(os.path.dirname(__file__), '%07d.jpg' % data['id']))
    print(filename)
