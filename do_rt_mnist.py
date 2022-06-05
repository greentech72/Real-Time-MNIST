# perform real-time selective search

# imports
import cv2
import time
import keras
import random
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# cfg
ACTUAL_FRAME_SIZE = 28 * 8

# load model
model = keras.models.load_model('MNIST_model.h5')

# video capture
cap = cv2.VideoCapture()
cap.open(0, cv2.CAP_DSHOW)
f = time.time()  # to calc FPS

while True:
	ret, frame = cap. read()  # get image from camera

	# calculate fps
	fps = round(1 / (time.time() - f) * 100) / 100
	f = time.time()

	# add fps to img
	cv2.putText(frame, 'FPS : ' + str(fps), (10, 30), 
		cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

	# calculate center of thee camera
	CENTER = (frame.shape[1] // 2 - ACTUAL_FRAME_SIZE // 2, frame.shape[0] // 2 - ACTUAL_FRAME_SIZE // 2)
	x1, y1, x2, y2 = CENTER[0], CENTER[1], CENTER[0] + ACTUAL_FRAME_SIZE, CENTER[1] + ACTUAL_FRAME_SIZE

	# show actual frame
	cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

	# get this frame
	tmp = frame[y1:y2, x1:x2].copy()

	# preprocess frame
	tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
	tmp = (tmp / tmp.max())
	tmp = tmp - tmp.min()
	tmp = (tmp * 255).astype('uint8')
	tmp = 1 - tmp
	tmp = np.where(tmp > 128, 255, 0).astype('uint8')

	# resize to 28x28
	tmp = np.array(Image.fromarray(tmp).resize((28, 28)))

	# convert to correct tensor
	tmp.shape = (1,) + tmp.shape + (1,)

	# predict class
	prediction = model.predict(tmp / 255.)[0].argmax()

	# show class
	cv2.putText(frame, ('none' if prediction == 10 else str(prediction)), (x1, y1 - 20), 
		cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
	
	# place mini frame
	frame[y1:y1+28, x1:x1+28] = tmp

	# show frame
	cv2.imshow('frame', frame)

	# for exit 'q'
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
