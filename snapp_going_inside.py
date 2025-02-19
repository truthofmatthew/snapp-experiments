import os
import requests
import base64
import time
import json
from torch_predict import get_prediction_function
import numpy as np  


BASE_URL = "https://app.snapp.taxi/api"


HEADERS = {
        "accept": "*/*",
        "content-type": "application/json",
        "app-version": "pwa",
        "x-app-name": "passenger-pwa",
        "x-app-version": "v18.14.2",
        "locale": "fa-IR",
        "origin": "https://app.snapp.taxi",
        "referer": "https://app.snapp.taxi/login",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
        }


PHONE_NUMBER = "+98123456789"  
INITIAL_PAYLOAD = {
        "cellphone": PHONE_NUMBER,
        "attestation": {
                "method": "skip",
                "platform": "skip"
                },
        "extra_methods": []
        }



def fix_base64_padding(image_data):
    padding_needed = len(image_data) % 4
    if padding_needed != 0:
        image_data += '=' * (4 - padding_needed)
    return image_data



def convert_to_serializable(data):
    if isinstance(data, (np.generic,)):  
        return data.item()
    elif isinstance(data, np.ndarray):  
        return data.tolist()
    elif isinstance(data, dict):  
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):  
        return [convert_to_serializable(item) for item in data]
    else:
        return data



def send_otp_request():
    
    url = f"{BASE_URL}/api-passenger-oauth/v3/mutotp"
    response = requests.post(url, headers=HEADERS, data=json.dumps(INITIAL_PAYLOAD))

    if response.status_code == 200:
        print("OTP sent successfully.")
        return True
    elif response.status_code == 401:
        print("Captcha required.")
        return handle_captcha()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False



def handle_captcha():
    
    captcha_url = f"{BASE_URL}/captcha/api/v1/generate/text/numeric/71C84A80-395B-448E-A240-B7DC939186D3"
    response = requests.get(captcha_url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to fetch captcha: {response.status_code} - {response.text}")
        return False

    try:
        captcha_data = response.json()
        image_data = captcha_data["image"].split(",")[1] if "," in captcha_data["image"] else captcha_data["image"]
        image_data = fix_base64_padding(image_data)
        image_bytes = base64.b64decode(image_data)

        
        timestamp = int(time.time())
        if not os.path.exists('captcha'):
            os.makedirs('captcha')
        image_file_name = f"{timestamp}_image.jpg"
        image_file_path = os.path.join('captcha', image_file_name)
        with open(image_file_path, "wb") as f:
            f.write(image_bytes)
        print(f"Captcha image saved as '{image_file_path}'")

        
        full_image_path = os.path.abspath(image_file_path)
        pred_text, confs, _ = get_prediction_function(full_image_path)
        print(f"Predicted CAPTCHA: {pred_text}, Confidences: {confs}")
        token = input("Enter the token received via SMS: ")

        
        captcha_payload = {
                "attestation": {
                        "method": "numeric",
                        "platform": "captcha"
                        },
                "grant_type": "sms_v2",
                "client_id": "Find it yourself",
                "client_secret": "Find it yourself",
                "cellphone": PHONE_NUMBER,
                "token": token,
                "referrer": "pwa",
                "device_id": "Find it yourself",
                "captcha": {
                        "client_id": "Find it yourself",
                        "solution": pred_text,
                        "ref_id": captcha_data["ref_id"],
                        "type": "numeric"
                        }
                }

        url = f"{BASE_URL}/api-passenger-oauth/v3/mutotp/auth"
        response = requests.post(url, headers=HEADERS, data=json.dumps(captcha_payload))

        if response.status_code == 200:
            print("Captcha solved successfully.")
            return True
        else:
            print(f"Captcha submission failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error while handling captcha: {e}")
        return False


if __name__ == "__main__":
    if send_otp_request():
        print("Process completed successfully.")
    else:
        print("Process failed.")