import requests, json, base64

url = "http://localhost:8181/predict"

image_path = "../image.jpeg"

result = requests.post(url, json={"image": base64.b64encode(open(image_path, "rb").read())}).text

print(json.loads(result))
