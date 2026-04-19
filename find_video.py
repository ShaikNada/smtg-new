import urllib.request
import json
import urllib.parse

query = urllib.parse.quote('robbery CCTV')
url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={query}&utf8=&format=json"
req = urllib.request.urlopen(url)
data = json.loads(req.read())

for item in data['query']['search']:
    print(item['title'])
