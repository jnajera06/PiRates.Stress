# -*- coding: utf-8 -*-
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Q10_1':3,
'Q10_4':2,
'Q10_5':3,
'Q10_6':2,
'Q10_7':3,
'Q10_9':2,
'Q10_12':2,
'Q17_1':3,
'Q17_7':2,
'Q17_10':2,
})

print(r.json())