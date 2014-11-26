import cv2
import numpy as np
import math
import platform
import utils

########################################################
# Background subtraction applied to object recognition #
########################################################

# Windows array
W = []
# Number of frames to sample before substraction happens
w = 20
# Learning rate
alpha = 0.1
# Minimum area of an object (in pixels^2)
min_area = 10000
# Centroids window size
k = (20, 20)

# Background initialization
BG = None

if __name__ == '__main__':

	numframes = 0
	capture = cv2.VideoCapture(0)
	if(platform.system() == "Darwin"):
		capture.set(3,300)
		capture.set(4,240)


	cv2.namedWindow('BSaOR', cv2.WINDOW_NORMAL)

	_,frame = capture.read()

	windowSize = frame.shape
	window = utils.Window(windowSize[0]*2, windowSize[1]*2)
	window.open('BSaOR')

	#window = np.zeros((frame.shape[0]*2, frame.shape[1]*2, frame.shape[2]))
	deiluminate = np.zeros((frame.shape[0], frame.shape[1], 1))
	substraction = np.zeros((frame.shape[0], frame.shape[1], 1))
	recognition = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
	applied = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))

	while True:
		_,frame = capture.read()
		frame = np.array(frame, np.float32) / 255.0
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		if BG is None:
			BG = gray
		else:
			W.append(gray)

			l = len(W)
			if l == w:
				difference = np.zeros(l)
				for i in range(0, l):
					difference[i] = sum(sum(BG-W[i]))

				# Compute which area changes the most compared to the backgound
				mostChanges = np.abs(sum(W-BG))
				preThreshold = mostChanges

				# Saturate
				mostChanges = (mostChanges.clip(0, max=1)*255).astype('uint8')

				# Apply a threshold
				_,mostChanges = cv2.threshold(mostChanges, 150, 255, cv2.THRESH_BINARY)
				mostChanges = cv2.medianBlur(mostChanges, 9)

				# Save image substaction
				substraction = mostChanges

				# BGR
				recognition[:,:,1] = mostChanges
				recognition[:,:,0] = np.zeros((mostChanges.shape[0], mostChanges.shape[1]))
				recognition[:,:,2] = np.zeros((mostChanges.shape[0], mostChanges.shape[1]))

				# Apply K windows of size k to find centroids
				centroids = []
				for i in xrange(1, mostChanges.shape[0], k[0]):
					for j in xrange(1, mostChanges.shape[1], k[1]):
						# Get centroid points
						points = np.argwhere(mostChanges[i:i+k[0],j:j+k[0]])

						# Find centroid
						x_coords = [i + p[0] for p in points]
						y_coords = [j + p[1] for p in points]
							
						_len = len(x_coords)
						if _len > 0:
							center = (sum(y_coords)/_len, sum(x_coords)/_len)

							# [x1, y1, x2, y2, (cx, cy)]
							rect = [(center[0] - k[1]/2, center[1] - k[0]/2),\
									(center[0] + k[1]/2, center[1] + k[0]/2),\
									center]

							# [x, y, w, h]
							ctr = (center[0] - k[1]/2, center[1] - k[0]/2, k[0], k[1])

							centroids.append(ctr)

							#cv2.circle(recognition, ctr[2], 5, (0, 0, 255), 5)
							cv2.rectangle(recognition, rect[0], rect[1], (255, 0, 0), 2)

				if centroids != []:
					boxes,_ = cv2.groupRectangles(centroids, 0)

					mask = np.zeros((frame.shape[0], frame.shape[1], 1), 'uint8')
					applied = np.zeros((frame.shape[0], frame.shape[1], 1))

					for b in boxes:
						area = mask[b[1]:b[1]+b[2],b[0]:b[0]+b[3]]
						mask[b[1]:b[1]+b[2],b[0]:b[0]+b[3]] = np.ones((area.shape[0],area.shape[1],1))

					mask = cv2.medianBlur(mask, 9)#Intentem que no quedin forats
					mask = np.multiply(gray, mask)
					mask = (mask*255).astype('uint8')
					contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

					#cv2.imshow('Applied', applied)
					#cv2.imshow('contours', mask)

					applied = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
					applied[:,:,0] = mask

					i = 0
					for c in contours:
						leftmost = c[:,:,0].min()
						rightmost = c[:,:,0].max()
						topmost = c[:,:,1].min()
						bottommost = c[:,:,1].max()

						area = ((rightmost - leftmost)*(bottommost-topmost))
						if area < min_area:
							continue

						if i%2 == 0:
							ch = np.array(applied[:,:,1])
							cv2.drawContours(ch, c, -1, (250), 5)
							applied[:,:,1] = ch
						else:
							ch = np.array(applied[:,:,2])
							cv2.drawContours(ch, c, -1, (250), 5)
							applied[:,:,2] = ch

						i = i + 1

						obj = gray[topmost:bottommost,leftmost:rightmost]
						cv2.imshow('Obj' + str(i), obj)

				m = difference.argmin()
				BG = (1-alpha)*BG + alpha * W[m]
				W = []

		# Set window parameters
		# Top-Left: Background
		window.showGrayAt((0, 0, windowSize[0], windowSize[1]), BG)

		# Top-Right
		applied = applied.squeeze()
		window.showAt((0, windowSize[1], windowSize[0], windowSize[1]*2), applied)

		#sums = [1*a0+2*a1+4*a3+8*a4+16*a5+32*a6+64*a7*128*a8 for a0,a1,a2,a3,a4,a6,a7,a8,a9]
		# Bottom left: Substraction
		window.showGrayAt((windowSize[0], 0, windowSize[0]*2, windowSize[1]), substraction)

		# Bottom right: 1st stage recognition
		window.showAt((windowSize[0], windowSize[1], windowSize[0]*2, windowSize[1]*2), recognition)

		window.text("Informacio:(esc per sortir)", (50,50), 1)
		window.text("Frames: " + str(numframes), (50,80), 1)

		window.render()

		key = cv2.waitKey(10)
		numframes += 1
		if key & 0xFF == 27:
			break

	cv2.destroyAllWindows()
	capture.release()
