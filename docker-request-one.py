import requests
import json

url = 'http://127.0.0.1:4000/predict'

data = {'country':'eire','year':'2018','month':'05','day':'01','mode':'test'}

j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

r = requests.post(url, data=j_data, headers=headers)

print(r, r.text)