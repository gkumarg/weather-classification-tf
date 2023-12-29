# Weather Classification Webapp

This project will help you discern the type of weather seen in the image. Upload the image and the model will classify the image as depicting one of these weather scenarios:
dew, fog/smog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, snow

# Deploy ML model with Flask to Heroku

Click the button below to quickly clone and deploy into your own Heroku acount.
If you don't have one it'll prompt you to setup a free one.

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

Once deployed to your Heroku instance run the following:

```bash
curl -s -XPOST 'https://<name-of-your-heroku-app>.herokuapp.com/' -d '{image_url:'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Rain_falling.JPG/450px-Rain_falling.JPG'}' -H 'accept-content: application/json'
```

Alternatively a simple python script:

```python
import requests
import json
url = 'https://<name-of-your-heroku-app>.herokuapp.com/'
image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Rain_falling.JPG/450px-Rain_falling.JPG'
response = requests.get(image_url)
image_data = BytesIO(response.content)
files = {'file': image_data}
response = requests.post(url, files=files)
data = response.json()
print("Image URL:", image_url)
print("Prediction:", data['prediction'])
```
