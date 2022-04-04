import json
import requests
from Image_Pipelines import DisplayImageToArray
import numpy as np
import glob
import matplotlib.pyplot as plt

#incase of proxies problem
os.environ ['no_proy'] = 'http://localhost:8501/v1/models/inception'
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

proxies = {'http':None,'https': None}

test = []
imgsize =224,224
for index,img in enumerate(glob.glob('test1/*.jpg')):
    test.append(DisplayImageToArray(img,imgsize))
    if index == 3:
        break


test = np.asarray(test,np.float32)

request_url = 'http://tensorflow-serving:8501/v1/models/inception:predict'
request_body = json.dumps({'signature_name':'serving_default','instances':test.tolist()})
json_headers = {'content_type':'application/json'}
json_response = requests.post(request_url,data = request_body,headers = json_headers,verify = False,proxies = proxies)
response_body =json.loads(json_response.text)
predictions = response_body['predictions']


#get the predicted values
for i in predictions:
    if np.argmax(i) == 1:
        print('1 --->dog')
    else:
        print('0 --->cat')
        
