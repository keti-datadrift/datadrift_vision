import requests
import json
from time import time

data = { 
	"instruct": "please describe this image"
	}

headers = {'Content-type': 'application/json'}
t0=time()
response = requests.post('http://locahost:8000/api/drift/v1/vlm_verify/', data=json.dumps(data), headers=headers)
t1=time()
print(t1-t0)
print('****************************************************************')
print('response.text=',response.text)