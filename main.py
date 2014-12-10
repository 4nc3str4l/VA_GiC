
########################################################
# Background subtraction applied to object recognition #
########################################################

import cv2
import numpy as np
import math
import platform
import utils
import cnn


########################################################
#				   	  CONSTANTS     				   #
########################################################
class Constants:
	# Number of frames to sample before substraction happens
	w = 20
	# Learning rate
	alpha = 0.1
	# Minimum area of an object (in pixels^2)
	min_area = 1500
	# Centroids window size
	k = (5, 5)

########################################################
#				   	    SYSTEM      				   #
########################################################
class System:
	# Frames array
	W = []
	# Convolutional Neural Network
	net = None
	# Background initialization
	BG = None


if __name__ == '__main__':
	########################################################
	#				   CAFFE LOADING     				   #
	########################################################
	bsaor_root = '/home/guillem/UB/Caffe/examples/bsaor/'
	cnn.load('/usr/local/caffe2/')
	#cnn.disable()
	System.net = cnn.init(
		bsaor_root + 'bsaor.prototxt', 
		bsaor_root + 'bsaor_iter_6000.caffemodel',
		cnn.mean(bsaor_root + 'bsaor_train_lmdb.binaryproto').reshape(3, 150, 150),
		raw_scale=255,
		image_dims=(150, 150),
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

		# First frame?
		if System.BG is None:
			System.BG = gray
		else:
			System.W.append(gray)

			l = len(System.W)
			if l == Constants.w:
				# Substract BG to current frame
				mostChanges = np.abs(sum(System.W-System.BG))
				
				# Normalize max-min values
				mostChanges = (mostChanges.clip(0, max=1)*255).astype('uint8')

				# Apply a threshold
				_,mostChanges = cv2.threshold(mostChanges, 150, 255, cv2.THRESH_BINARY)
				mostChanges = cv2.medianBlur(mostChanges, 9)

				# BGR
				recognition[:,:,1] = mostChanges
				recognition[:,:,0] = np.zeros((mostChanges.shape[0], mostChanges.shape[1]))
				recognition[:,:,2] = np.zeros((mostChanges.shape[0], mostChanges.shape[1]))

				# So... yeah
				mask = cv2.medianBlur(mostChanges, 3)
				mask = np.multiply(gray, mask)
				mask = (mask*255).astype('uint8')

				# Filter out distance
				mask = cv2.distanceTransform(mask, cv2.cv.CV_DIST_L1, 3)
				mask = cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX).astype('uint8')
				mask = cv2.GaussianBlur(mask, (-3,-3), 0.5)
				_,mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

				#cv2.imshow('test', mask)

				contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

				applied = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
				applied[:,:,0] = mask

				colors = (\
						(0, 100, 0),\
						(0, 0, 100),\
						(0, 100, 100),\
						(100, 100, 100),\
						(100, 0, 100),\
						(255, 255, 255),\
					)
				objs = []
				i = 0
				for c in contours:
					leftmost = c[:,:,0].min()
					rightmost = c[:,:,0].max()
					topmost = c[:,:,1].min()
					bottommost = c[:,:,1].max()

					area = ((rightmost - leftmost)*(bottommost-topmost))
					if area < Constants.min_area:
						continue

					for chi in range(0, 3):
						ch = np.array(applied[:,:,chi])
						cv2.drawContours(ch, c, -1, colors[i % len(colors)][chi], 5)
						applied[:,:,chi] = ch

					i = i + 1

					obj = frame[topmost:bottommost,leftmost:rightmost,:]
					cv2.imshow('Obj' + str(i), obj)
					objs.append(obj)

				if objs != []:
					prediction = System.net.predict(objs)
					print prediction

				# Compute difference of each frame with respect to BG
				difference = np.zeros(l)
				for i in range(0, l):
					difference[i] = sum(sum(System.BG-System.W[i]))

				# Get minimum and set new BG
				m = difference.argmin()
				System.BG = (1-Constants.alpha)*System.BG + Constants.alpha * System.W[m]
				System.W = []

				# Set window parameters
				# Top-Left: Background
				window.showGrayAt((0, 0, windowSize[0], windowSize[1]), System.BG)

				# Top-Right: Substraction (mostChanges)
				window.showGrayAt((0, windowSize[1], windowSize[0], windowSize[1]*2), mostChanges)

				# Bottom-Left: Recognition
				window.showAt((windowSize[0], 0, windowSize[0]*2, windowSize[1]), recognition)

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
