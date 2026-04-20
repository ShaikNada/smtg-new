import requests
import json

url = "http://127.0.0.1:8000/api/ocr-preview"
file_path = r"C:\Users\Nazeer\Desktop\another\smtg-new\test_fir.png" # Using an existing test image since I can't easily access the generated one's path in this script without hardcoding

# Let's try to find the generated image path from the previous tool output
# Generated image: C:\Users\Nazeer\.gemini\antigravity\brain\98b753b0-4190-4b55-b09b-fd128b363801\test_fir_document_1776671563095.png

gen_file_path = r"C:\Users\Nazeer\.gemini\antigravity\brain\98b753b0-4190-4b55-b09b-fd128b363801\test_fir_document_1776671563095.png"

# First, we need to be logged in to access the API
session = requests.Session()
login_url = "http://127.0.0.1:8000/login"
session.post(login_url, data={"username": "admin", "password": "admin123"})

with open(gen_file_path, "rb") as f:
    files = {"fir_image": f}
    response = session.post(url, files=files)

print(json.dumps(response.json(), indent=2))
