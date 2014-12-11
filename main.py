
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

class Constants:
	labels = ('Face', 'Suitcase', 'Black', 'Blue', 'Gray', 'Yellow')
	colors = (\
		(0, 1, 0),\
		(0, 0, 1),\
		(0.10, 0.10, 0.10),\
		(1, 0, 0),\
		(0.5, 0.5, 0.5),\
		(0, 1, 1),\
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
		'models/bsaor/bsaor_iter_3000.caffemodel',
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
		_,frame = capture.read()
		frame = np.array(frame, np.float32) / 255.0
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		applied = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
		update, objs = System.rbg.feed(gray)
		if objs != []:
			ims = []

			applied[:,:,0] = System.rbg.getMask()

			for obj in objs:
				im = frame[obj[2]:obj[3],obj[0]:obj[1],:]
				ims.append(im)

			prediction = System.net.predict(ims)

			i = 0
			window.forget_text()
			for p in prediction:
				idx = np.argmax(p)
				name = Constants.labels[idx]
				print "[%06.2f%%] %s" % (p[idx] * 100, name)

				p = np.delete(p, idx)
				idx2 = np.argmax(p)
				idxlabel = idx2 if idx2 < idx else idx2 + 1
				second = Constants.labels[idxlabel]
				print "\t[%06.2f%%] %s" % (p[idx2] * 100, second)

				for chi in range(0, 3):
					ch = np.array(applied[:,:,chi])
					cv2.rectangle(ch, (objs[i][0], objs[i][2]), (objs[i][1], objs[i][3]), Constants.colors[idx][chi], 5)
					applied[:,:,chi] = ch

					window.perm_text(name, (windowSize[1] + objs[i][0] - 1, windowSize[0] + objs[i][2] - 11), 1, color=(0,0,0), thickness=3)
					window.perm_text(name, (windowSize[1] + objs[i][0], windowSize[0] + objs[i][2] - 10), 1, thickness=2)

				i = i + 1

				name = 'Obj' + str(i)
				cv2.namedWindow(name, cv2.WINDOW_NORMAL)
				cv2.resizeWindow(name, 100, 100)
				cv2.moveWindow(name, 50 + i*100 + i*10, 50)
				cv2.imshow(name, ims[i - 1])
			print				

			for k in range(i + 1, System.wi):
				cv2.destroyWindow('Obj' + str(k))

			System.wi = i + 1

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


		window.text("Informacio:(esc per sortir)", (50,50), 1)
		window.text("Frames: " + str(numframes), (50,80), 1)

		window.render()

		key = cv2.waitKey(10)
		numframes += 1
		if key & 0xFF == 27:
			break

	cv2.destroyAllWindows()
	capture.release()
