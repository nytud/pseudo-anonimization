import requests
import json

r = requests.post('http://127.0.0.1:5000/tok/morph', data={'text': 'Ilonával'})
resp = r.text.split("\t")[-1]
info = json.loads(resp)
lemma = info[0]["lemma"]
 
print(info[0]["tag"])