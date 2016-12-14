import os
import json
from urllib.request import urlopen, urlretrieve, Request

url = os.environ['API_ENDPOINT'] + '/faces/random.json'
auth_headers = {
    'X-User-Email': os.environ['API_AUTH_EMAIL'],
    'X-User-Token': os.environ['API_AUTH_TOKEN'],
}
req = Request(url, None, auth_headers)
for i in range(100):
    data = json.loads(urlopen(req).read().decode())
    filename, _ = urlretrieve(data['image_url'], os.path.join(os.path.dirname(__file__), 'images', '%07d.jpg' % data['id']))
    print(filename)
