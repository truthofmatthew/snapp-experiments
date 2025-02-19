import base64
import json
import os
import time

import numpy as np
import requests

from torch_predict import get_prediction_function


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


url = "https://app.snapp.taxi/api/captcha/api/v1/generate/text/numeric/71C84A80-395B-448E-A240-B7DC939186D3"

headers = {
        "Accept": "image/webp,*/*",
        "Authorization": "Bearer your_token_here",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://app.snapp.taxi/",
        "Host": "app.snapp.taxi"
        }

if not os.path.exists('captcha'):
    os.makedirs('captcha')

timestamp = int(time.time())

response = requests.get(url, headers=headers)

if response.status_code == 200:
    try:

        image_data = response.json().get("image")

        if not image_data:
            raise ValueError("No 'image' field found in the response")

        image_data = image_data.split(",")[1] if "," in image_data else image_data

        image_data = image_data.strip()

        image_data = fix_base64_padding(image_data)

        image_bytes = base64.b64decode(image_data)

        image_file_name = f"{timestamp}_image.jpg"
        image_file_path = os.path.join('captcha', image_file_name)
        with open(image_file_path, "wb") as f:
            f.write(image_bytes)
        print(f"Image saved as '{image_file_path}'")

        response_file_name = f"{timestamp}_response.json"
        response_file_path = os.path.join('captcha', response_file_name)
        with open(response_file_path, "w") as f:
            json.dump(response.json(), f, indent=4)
        print(f"Response saved as '{response_file_path}'")

        full_image_path = os.path.abspath(image_file_path)
        pred_text, confs, _ = get_prediction_function(full_image_path)

        prediction_data = {
                "predicted_text": pred_text,
                "confidences": convert_to_serializable(confs),
                "other_data": convert_to_serializable(_)
                }

        prediction_file_name = f"{timestamp}_prediction.json"
        prediction_file_path = os.path.join('captcha', prediction_file_name)
        with open(prediction_file_path, "w") as f:
            json.dump(prediction_data, f, indent=4)
        print(f"Prediction saved as '{prediction_file_path}'")

        print(f"Predicted CAPTCHA: {pred_text}, Confidences: {confs}")
    except base64.binascii.Error as e:
        print(f"Error while decoding base64: {e}")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
else:
    print(f"Failed to retrieve image. Status code: {response.status_code}")
