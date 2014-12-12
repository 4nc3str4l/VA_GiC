
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
########################################################
class System:
	# Convolutional Neural Network
	net = None

	# Running background
	rbg = RunningBackground(20, 0.1, 500)

	# Last window
	wi = 0


########################################################
#				   	  CONSTANTS      				   #
########################################################
class Constants:
	labels = ('Face', 'Suitcase', 'Black', 'Blue', 'Gray', 'Yellow', 'Skin', 'Purple')
	colors = (\
		(0, 1, 0),\
		(0, 0, 1),\
		(0.10, 0.10, 0.10),\
		(1, 0, 0),\
		(0.5, 0.5, 0.5),\
		(0, 1, 1),\
		(0, .5, .5),\
		(.5, 0, .5),\
	)


if __name__ == '__main__':
	########################################################
	#				   CAFFE LOADING     				   #
	########################################################
	bsaor_root = '/home/guillem/UB/Caffe/examples/bsaor/'
	cnn.load('/usr/local/caffe2/')
	#cnn.disable()
	System.net = cnn.init(
		'models/bsaor/bsaor.prototxt', 
		'models/bsaor/bsaor_iter_15000.caffemodel',
		cnn.mean('models/bsaor/bsaor_train_lmdb.binaryproto').reshape(3, 100, 100),
		raw_scale=255,
		image_dims=(95, 95),
		gpu=True
	)

	########################################################
	#				    CAMERA POOLING					   #
	########################################################

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
		# Capture frame, convert to float and to grayscale
		_,frame = capture.read()
		frame = np.array(frame, np.float32) / 255.0
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		# Create a matrix of zeros to hold the segmenation+recognition visualization
		applied = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))

		# Feed the frame to our background
		update, objs = System.rbg.feed(gray)

		# Has it segmented any objects?
		if objs != []:
			ims = []

			applied[:,:,0] = System.rbg.getMask()

			# Foreach object, get the real image
			for obj in objs:
				im = frame[obj[2]:obj[3],obj[0]:obj[1],:]
				ims.append(im)

			# Predict result with our CNN
			prediction = System.net.predict(ims)

			# Foreach prediction, show class (label)
			i = 0
			window.forget_text()
			for p in prediction:
				# Most probable class
				idx = np.argmax(p)
				name = Constants.labels[idx]
				print "[%06.2f%%] %s" % (p[idx] * 100, name)

				# 2nd most probable class
				p = np.delete(p, idx)
				idx2 = np.argmax(p)
				idxlabel = idx2 if idx2 < idx else idx2 + 1
				second = Constants.labels[idxlabel]
				print "\t[%06.2f%%] %s" % (p[idx2] * 100, second)

				# Bounding box of the object, on all 3 BGR channels
				for chi in range(0, 3):
					ch = np.array(applied[:,:,chi])
					cv2.rectangle(ch, (objs[i][0], objs[i][2]), (objs[i][1], objs[i][3]), Constants.colors[idx][chi], 5)
					applied[:,:,chi] = ch

					# Permanent text showing the class
					window.perm_text(name, (windowSize[1] + objs[i][0] - 1, windowSize[0] + objs[i][2] - 11), 1, color=(0,0,0), thickness=3)
					window.perm_text(name, (windowSize[1] + objs[i][0], windowSize[0] + objs[i][2] - 10), 1, thickness=2)

				# Window (and object) index
				i = i + 1

				# Show window
				name = 'Obj' + str(i)
				cv2.namedWindow(name, cv2.WINDOW_NORMAL)
				cv2.resizeWindow(name, 100, 100)
				cv2.moveWindow(name, 50 + i*100 + i*10, 50)
				cv2.imshow(name, ims[i - 1])
			print				

			# Destroy all windows of previous objects which no longer exist
			for k in range(i + 1, System.wi):
				cv2.destroyWindow('Obj' + str(k))

			# Save current number of windows
			System.wi = i + 1

		# If there has been an update (segmentation), update window matrix
		if update:
			# Set window parameters
			# Top-Left: Background
			window.showGrayAt((0, 0, windowSize[0], windowSize[1]), System.rbg.getBG())

			# Top-Right: Recognition
			window.showGrayAt((0, windowSize[1], windowSize[0], windowSize[1]*2), 255-System.rbg.getRecognition())

			# Bottom-Left: Substraction (mostChanges)
			window.showGrayAt((windowSize[0], 0, windowSize[0]*2, windowSize[1]), System.rbg.getMostChanges())

			# Bottom-Right: Object segmentation
			window.showAt((windowSize[0], windowSize[1], windowSize[0]*2, windowSize[1]*2), applied)


		# Print current frames
		window.text("Press [ESC] to exit", (50,50), 1)
		window.text("Frames: " + str(numframes), (50,80), 1)

		# Render window
		window.render()

		# Check if [ESC] is pressed
		key = cv2.waitKey(10)
		numframes += 1
		if key & 0xFF == 27:
			break

	# Destroy all windows and release camera
	cv2.destroyAllWindows()
	capture.release()
