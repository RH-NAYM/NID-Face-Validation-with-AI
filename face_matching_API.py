import requests
from PIL import Image, ImageDraw
import numpy as np
import face_recognition
from io import BytesIO
import base64 
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import logging
import pytz
from datetime import datetime
import uvicorn
import zipfile
import os

logging.basicConfig(filename="faceMatching.log", filemode='w')
logger = logging.getLogger("Face")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("faceMatching.log")
logger.addHandler(file_handler)
total_done = 0
total_error = 0

FACE_RECOGNITION_TOLERANCE = 0.6

def get_bd_time():
    bd_timezone = pytz.timezone("Asia/Dhaka")
    time_now = datetime.now(bd_timezone)
    current_time = time_now.strftime("%I:%M:%S %p")
    return current_time

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def compare_faces_from_urls(image_url_1, image_url_2, tolerance=FACE_RECOGNITION_TOLERANCE):
    def get_face_info_from_url(image_url):
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image = np.array(image)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                face_locations = [(top, right, bottom, left) for (top, right, bottom, left) in face_locations]
                top, right, bottom, left = face_locations[0]
                face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                img_with_rectangles = Image.fromarray(image)
                draw = ImageDraw.Draw(img_with_rectangles)
                draw.rectangle([left-20, top-50, right+20, bottom], outline="green", width=5)

                return face_encoding, img_with_rectangles
        return None, None
    face_encoding_1, img_with_rectangles_1 = get_face_info_from_url(image_url_1)
    face_encoding_2, img_with_rectangles_2 = get_face_info_from_url(image_url_2)

    if face_encoding_1 is not None and face_encoding_2 is not None:
        are_same_person = face_recognition.compare_faces([face_encoding_1], face_encoding_2, tolerance=tolerance)[0]

        if are_same_person:
            result = "Same Person"
        else:
            result = "Different Person"
        img1_base64 = pil_image_to_base64(img_with_rectangles_1)
        img2_base64 = pil_image_to_base64(img_with_rectangles_2)

        return result, img1_base64, img2_base64

    return "No Face Detected", None, None

app = FastAPI()

class Item(BaseModel):
    img1: str
    img2: str

async def process_item(item: Item):
    # print("asdasda",item)
    try:
        result, img1_base64, img2_base64 = compare_faces_from_urls(item.img1, item.img2)
        return {"AI": result, "img1": img1_base64, "img2": img2_base64}
    except Exception as e:
        return {"AI": f"Error: {str(e)}"}

@app.post("/face")
async def create_items(items: Item):
    # print(items)
    try:
        result = await process_item(items)
        r= {"AI":result["AI"]}
        p = {"Payload":items}
        print(r)
        print(p)
        return result
    except Exception as e:
        global total_error
        total_error += 1
        logger.info(f"Time:{get_bd_time()}, Execution Failed and Total Failed Execution : {total_error}, Payload:{items}")
        logger.error(str(e))
        return {"AI": f"Error: {str(e)}"}
    finally:
        global total_done
        total_done += 1
        logger.info(f"Time:{get_bd_time()}, Execution Done and Total Successful Execution : {total_done}, Payload:{items}")

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=8060)
    except Exception as e:
        print(f"Server error: {str(e)}")
        pass
