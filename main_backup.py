import cv2
import numpy as np
import math
from itertools import izip_longest


def grouper(iterable, n, fillvalue=None):
	args = [iter(iterable)] * n
	return izip_longest(*args, fillvalue=fillvalue)

# Background subtraction applied to object recognition

def writeText(text, pos, size, frame):
	cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (255,255,255))



W = []
w = 20
alpha = 0.1
BG = None

if __name__ == '__main__':

	numframes = 0
	capture = cv2.VideoCapture(0)
	cv2.namedWindow('BSaOR', cv2.WINDOW_NORMAL)

	_,frame = capture.read()

	windowSize = frame.shape

	window = np.zeros((frame.shape[0]*2, frame.shape[1]*2, frame.shape[2]))
	deiluminate = np.zeros((frame.shape[0], frame.shape[1], 1))
	substraction = np.zeros((frame.shape[0], frame.shape[1], 1))
	recognition = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))

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

				# Apply K windows of size k
				centroids = []
				k = (50, 50)
				mi = int(math.floor(mostChanges.shape[0] / k[0]))
				mj = int(math.floor(mostChanges.shape[1] / k[1]))
				for i in xrange(1, mostChanges.shape[0], k[0]):
					for j in xrange(1, mostChanges.shape[1], k[1]):
						# Find centroid
						points = [p for p in zip(xrange(i,i+k[0]), xrange(j,j+k[1])) if p[0] < mostChanges.shape[0] and p[1] < mostChanges.shape[1] and mostChanges[p[0], p[1]] == 255]
						x_coords = [p[0] for p in points]
						y_coords = [p[1] for p in points]
							
						_len = len(x_coords)
						if _len > 0:
							center = (sum(y_coords)/_len, sum(x_coords)/_len)
							ctr = [(center[0] - k[1]/2, center[1] - k[0]/2),\
									(center[0] + k[1]/2, center[1] + k[0]/2),\
									center]

							centroids.append(ctr)

							cv2.circle(recognition, ctr[2], 5, (0, 0, 255), 5)
							cv2.rectangle(recognition, ctr[0], ctr[1], (255, 0, 0), 2)

				found = 1
				while found > 0 and centroids != []:
					found = 0
					origin = centroids[0]
					centroids.pop(0)

					distances = [(np.linalg.norm( (p[c2][0]-origin[c1][0],p[c2][1]-origin[c1][1]) ),p) for p in centroids for c2 in range(2, len(p)) for c1 in range(2, len(origin))]
					if distances == []:
						centroids.append(origin)
						break
					
					minDist = min(distances)

					if minDist[0] < k[0]*2:
						found = found + 1
						centroid = minDist[1]
						centroids.remove(centroid)

						left = min((origin[0][0], centroid[0][0]))
						top = min((origin[0][1], centroid[0][1]))
						right = max((origin[1][0], centroid[1][0]))
						bot = max((origin[1][1], centroid[1][1]))

						# Delete all centroids inside
						copy = centroids[::]
						for e in copy:
							centers = e[2:]
							for c in centers:
								if left < c[0] and right > c[0] and top < c[1] and bot > c[1]:
									centroids.remove(e)
									break

						ctr = [(left, top), (right, bot)] + origin[2:] + centroid[2:] + [((left+right)/2, (left+right)/2)]
						centroids.append(ctr)
					else:
						centroids.append(origin)

				copy = centroids[::]
				for ctr1 in copy:
					for ctr2 in copy:
						if ctr1 == ctr2:
							continue



				for ctr in centroids:
					p1 = np.array(ctr[0]).clip(0)
					p1 = (p1[0], p1[1])

					p2 = np.array(ctr[1]).clip(0)
					p2 = (p2[0], p2[1])

					print "At:",p1,p2

					cv2.rectangle(recognition, p1, p2, (255, 255, 255), 1)
				
				print

				m = difference.argmin()
				BG = (1-alpha)*BG + alpha * W[m]
				W = []

		# Set window parameters
		# Top-Left: Background
		window[0:windowSize[0], 0:windowSize[1], 0] = BG;
		window[0:windowSize[0], 0:windowSize[1], 1] = BG;
		window[0:windowSize[0], 0:windowSize[1], 2] = BG;
		# Top right: Deiluminate

		# This is way WAY WAAAY to slow
		"""
		# Window size is 3x3 ~ 8 bits ~ 255 intensity
		g = [np.dot((gray[i-1:i+2,j-1:j+2] < gray[i,j]).flatten(), (1,2,4,8,0,16,32,64,128)) \
				for i in xrange(1, gray.shape[0]-1) \
				for j in xrange(1, gray.shape[1]-1)]
		g = np.array(g, np.uint8).reshape((gray.shape[0]-2, gray.shape[1]-2))

		min_ = np.min(np.min(g))
		max_ = np.max(np.max(g))
		g = (g-min_)/(max_-min_)

		window[1:windowSize[0]-1, windowSize[1]+1:windowSize[1]*2-1, 0] = g;
		window[1:windowSize[0]-1, windowSize[1]+1:windowSize[1]*2-1, 1] = g;
		window[1:windowSize[0]-1, windowSize[1]+1:windowSize[1]*2-1, 2] = g;
		
		[
			[
				[ 0.34901962  0.44313726  0.51372552]
  				[ 0.33725491  0.43529412  0.49803922]
  			]
 			[
 				[ 0.30588236  0.4509804   0.50588238]
  				[ 0.30980393  0.44705883  0.50196081]
			]
		]
		"""

		#sums = [1*a0+2*a1+4*a3+8*a4+16*a5+32*a6+64*a7*128*a8 for a0,a1,a2,a3,a4,a6,a7,a8,a9]
		# Bottom left: Substraction
		window[windowSize[0]:windowSize[0]*2, 0:windowSize[1], 0] = substraction.squeeze()
		window[windowSize[0]:windowSize[0]*2, 0:windowSize[1], 1] = substraction.squeeze()
		window[windowSize[0]:windowSize[0]*2, 0:windowSize[1], 2] = substraction.squeeze()
		# Bottom right: 1st stage recognition
		window[windowSize[0]:windowSize[0]*2, windowSize[1]:windowSize[1]*2] = recognition

		currentWindow = window.copy()
		writeText("Informacio:(esc per sortir)", (50,50), 1, currentWindow)
		writeText("Frames: " + str(numframes), (50,80), 1, currentWindow)

		cv2.imshow('BSaOR', currentWindow)

		k = cv2.waitKey(10)
		numframes += 1
		if k & 0xFF == 27:
			break

	cv2.destroyAllWindows()
	capture.release()