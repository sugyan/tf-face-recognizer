from flask import Flask, jsonify, request
from tensorflow.models.image.cifar10 import cifar10
import tensorflow as tf
import tempfile
import subprocess
import json

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def api():
    f = request.files['image']
    tmp = tempfile.NamedTemporaryFile()
    f.save(tmp.name)
    output = subprocess.check_output(['python', 'recognize.py', '--batch_size', '1', '--image_file', tmp.name])
    result = json.loads(output)
    return jsonify(result=result)

if __name__ == '__main__':
    app.run()
