import requests 
from io import BytesIO

def test_predict():
    # Testing with file
    url = 'http://127.0.0.1:5000/predict'
    # files = {'file': open('/mnt/c/Users/gkuma/datascience/weather_classification_dl/data/sandstorm/2908.jpg', 'rb')}
    # response = requests.post(url, files=files)
    
    # Testing with URL
    # image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Rain_falling.JPG/450px-Rain_falling.JPG'
    image_url = 'https://www.canr.msu.edu/contentAsset/image/6898d064-63c6-4ab5-9366-b9ce84cfd406/fileAsset/filter/Resize,Jpeg/resize_w/750/jpeg_q/80'
    
    # Download the image
    response = requests.get(image_url)
    # Open the image file object in binary mode
    image_data = BytesIO(response.content)

    # Send a post request to the server with the image
    files = {'file': image_data}
    response = requests.post(url, files=files)


    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    print("Image URL:", image_url)
    print("Prediction:", data['prediction'])

if __name__ == "__main__":
    test_predict()
