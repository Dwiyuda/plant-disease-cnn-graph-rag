import urllib.request
import os

urllib.request.urlretrieve(
    'https://unpkg.com/vis-network/standalone/umd/vis-network.min.js',
    'vis-network.min.js'
)
print('Done, size:', os.path.getsize('vis-network.min.js'), 'bytes')