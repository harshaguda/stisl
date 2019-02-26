# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to out input directory of images")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])
image_path = args["images"]

orig = cv2.imread(image_path)
 
	# pre-process our image by converting it from BGR to RGB channel
	# ordering (since our Keras mdoel was trained on RGB ordering),
	# resize it to 64x64 pixels, and then scale the pixel intensities
	# to the range [0, 1]
image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0

image = img_to_array(image)
image = np.expand_dims(image, axis=0)
 
	# make predictions on the input image
pred = model.predict(image)
pre = pred.argmax(axis=1)[0]

print(pre)
#print(pred.argmax(axis=1)[1])