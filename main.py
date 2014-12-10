
########################################################
# Background subtraction applied to object recognition #
########################################################

import cv2
import numpy as np
import math
import platform
import utils
import cnn
from background import RunningBackground 


########################################################
#				   	    SYSTEM      				   #
##########,
##############################################
class System:
	# Convolutional Neural Network
	net = None

	# Running background
	rbg = RunningBackground(20, 0.1, 1500)

	# Last window
	wi = 0

if __name__ == '__main__':
	########################################################
	#				   CAFFE LOADING     				   #
	########################################################
	bsaor_root = '/home/guillem/UB/Caffe/examples/bsaor/'
	cnn.load('/usr/local/caffe2/')
	#cnn.disable()
	System.net = cnn.init(
		bsaor_root + 'bsaor.prototxt', 
		bsaor_root + 'bsaor_iter_2000.caffemodel',
		cnn.mean(bsaor_root + 'bsaor_train_lmdb.binaryproto').reshape(3, 100, 100),
		raw_scale=255,
		image_dims=(100, 100),
		gpu=True
	)

	# Setup some variables
	numframes = 0
	capture = cv2.VideoCapture(0)

	# Set capture size (force it?)
	if(platform.system() == "Darwin"):
		capture.set(3,300)
		capture.set(4,240)

	_,frame = capture.read()

	# Setup window
	windowSize = frame.shape
	window = utils.Window(windowSize[0]*2, windowSize[1]*2)
	window.open('BSaOR')

	# Setup some buffers
	deiluminate = np.zeros((frame.shape[0], frame.shape[1], 1))
	recognition = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))

	# Frame fetching loop
	while True:
		_,frame = capture.read()
		frame = np.array(frame, np.float32) / 255.0
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		objs = System.rbg.feed(gray)
		if objs != []:
			ims = []

			colors = (\
					(0, 100, 0),\
					(0, 0, 100),\
					(0, 100, 100),\
					(100, 100, 100),\
					(100, 0, 100),\
					(255, 255, 255),\
				)

			applied = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
			applied[:,:,0] = System.rbg.getMask()

			i = 0
			for obj in objs:
				for chi in range(0, 3):
					ch = np.array(applied[:,:,chi])
					cv2.rectangle(ch, (obj[0], obj[2]), (obj[1], obj[3]), colors[i % len(colors)][chi], 5)
					applied[:,:,chi] = ch

				i = i + 1

				im = frame[obj[2]:obj[3],obj[0]:obj[1],:]
				ims.append(im)

				name = 'Obj' + str(i)
				cv2.namedWindow(name, cv2.WINDOW_NORMAL)
				cv2.resizeWindow(name, 100, 100)
				cv2.moveWindow(name, 50 + i*100 + i*10, 50)
				cv2.imshow(name, im)
				

			for k in range(i + 1, System.wi):
				cv2.destroyWindow('Obj' + str(k))

			System.wi = i + 1

			prediction = System.net.predict(ims)
			print prediction

			# Set window parameters
			# Top-Left: Background
			window.showGrayAt((0, 0, windowSize[0], windowSize[1]), System.rbg.getBG())

			# Top-Right: Substraction (mostChanges)
			#window.showGrayAt((0, windowSize[1], windowSize[0], windowSize[1]*2), mostChanges)

			# Bottom-Left: Recognition
			#window.showAt((windowSize[0], 0, windowSize[0]*2, windowSize[1]), recognition)

			# Bottom-Right: Object segmentation
			window.showAt((windowSize[0], windowSize[1], windowSize[0]*2, windowSize[1]*2), applied)


		window.text("Informacio:(esc per sortir)", (50,50), 1)
		window.text("Frames: " + str(numframes), (50,80), 1)

		window.render()

		key = cv2.waitKey(10)
		numframes += 1
		if key & 0xFF == 27:
			break

	cv2.destroyAllWindows()
	capture.release()
