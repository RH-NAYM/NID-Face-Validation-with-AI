import base64
from PIL import Image
from io import BytesIO
img1_base64 = ""
img2_base64 = ""

img1_data = base64.b64decode(img1_base64)
img2_data = base64.b64decode(img2_base64)

img1 = Image.open(BytesIO(img1_data))
img2 = Image.open(BytesIO(img2_data))

img1.save("decoded_image1.jpg")
img2.save("decoded_image2.jpg")

