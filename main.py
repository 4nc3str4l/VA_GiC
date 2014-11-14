import cv2
import numpy as np
import math

# Background subtraction applied to object recognition

def writeText(text, pos, size, frame):
	cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (255,255,255))

W = []
w = 15
alpha = 0.5
BG = None

if __name__ == '__main__':

	numframes = 0
	capture = cv2.VideoCapture(0)
	cv2.namedWindow('image')

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

				r = np.zeros((mostChanges.shape[0], mostChanges.shape[1],3))
				r[:,:,1] = mostChanges # BGR

				# Apply K windows of size k
				k = (50, 50)
				mi = int(math.floor(mostChanges.shape[0] / k[0]))
				mj = int(math.floor(mostChanges.shape[1] / k[1]))
				for i in range(1, mostChanges.shape[0], k[0]):
					for j in range(1, mostChanges.shape[1], k[1]):
						lastPosition = (-1, -1)
						position = (i, j)

						# Find centroid
						while position != lastPosition:
							lastPosition = position
							sumX, sumY = 0, 0
							points = 0

							#for x in range(position[0], position[0] + k[0]):
								#for y in range(position[1], position[1] + k[1]):
							for x in range(i, i + k[0]):
								for y in range(j, j + k[1]):
									if x >= mostChanges.shape[0] or y >= mostChanges.shape[1]:
										continue;

									if mostChanges[x, y] == 255:
										sumX += x
										sumY += y
										points = points + 1

							if points > 0:
								position = (int(sumX / points), int(sumY / points))
							else:
								position = (-1, -1)

						if position != (-1, -1):
							cv2.circle(r, (position[1], position[0]), 5, (0, 0, 255), 5)
							cv2.rectangle(r, (position[1] - k[1]/2, position[0] - k[0] / 2), (position[1] + k[1]/2, position[0] + k[0] / 2), (255, 0, 0), 2)

						#print position

				cv2.imshow('Threshold', r)

				m = difference.argmin()
				#print m,V
				#print
				
				BG = (1-alpha)*BG + alpha * W[m]
				#BG = BG - np.min(min(BG))
				W = []

		t = BG.copy()
		writeText("Informacio:(esc per sortir)", (50,50), 1, t)
		writeText("Frames: " + str(numframes), (50,80), 1, t)
		cv2.imshow('Background', t)

		subs = gray-BG
		cv2.imshow('subs', subs)

		#_,th = cv2.threshold(subs, 0.05, 1, cv2.THRESH_BINARY)
		#cv2.imshow('threshold', th)

		k = cv2.waitKey(10)
		numframes += 1
		if k & 0xFF == 27:
			break

	cv2.destroyAllWindows()
	capture.release()
