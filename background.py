import cv2
import numpy as np
import math

class RunningBackground:
	
	def __init__(self, frameSampling, alpha, min_area):
		# Background initialization
		self.BG = None
		
		# Frames array
		self.W = []

		# Sample every x frames
		self.frameSampling = frameSampling

		# Learning rate
		self.alpha = alpha

		# Minimum area of an object (in pixels^2)
		self.min_area = min_area

		# Visualization
		self.mask = None

	def getBG(self):
		return self.BG

	def getMask(self):
		return self.mask

	def feed(self, gray):
		# assert(len(gray.shape) == 2 || gray.shape[2] == 1)
		objs = []
		self.W.append(gray)

		if self.BG is None:
			self.BG = gray
		else:
			l = len(self.W)
			if l == self.frameSampling:
				objs = self.__segment(gray)

				# Compute difference of each frame with respect to BG
				difference = np.zeros(l)
				for i in range(0, l):
					difference[i] = sum(sum(self.BG-self.W[i]))

				# Get minimum and set new BG
				m = difference.argmin()
				self.BG = (1-self.alpha)*self.BG + self.alpha * self.W[m]
			
				self.W = []

		return objs


	def __segment(self, gray):
		# Substract BG to current frame
		mostChanges = np.abs(sum(self.W-self.BG))
		
		# Normalize max-min values
		mostChanges = (mostChanges.clip(0, max=1)*255).astype('uint8')

		# Apply a threshold
		_,mostChanges = cv2.threshold(mostChanges, 150, 255, cv2.THRESH_BINARY)
		mostChanges = cv2.medianBlur(mostChanges, 9)

		# Apply median filter and multiply with the current scene
		mask = cv2.medianBlur(mostChanges, 3)
		mask = np.multiply(gray, mask)
		mask = (mask*255).astype('uint8')

		# Apply distance filter to minimize possible collisions
		mask = cv2.distanceTransform(mask, cv2.cv.CV_DIST_L1, 3)
		# Normalize between 0 and 255
		mask = cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX).astype('uint8')
		# Reduce noise
		mask = cv2.GaussianBlur(mask, (-3,-3), 0.5)
		# Apply threshold again to mask out real low values 
		# (Almost black zones, thus possible unions between objects)
		_,mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

		self.mask = mask.copy()

		contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		objs = []
		for c in contours:
			leftmost = c[:,:,0].min()
			rightmost = c[:,:,0].max()
			topmost = c[:,:,1].min()
			bottommost = c[:,:,1].max()

			area = ((rightmost - leftmost)*(bottommost-topmost))
			if area < self.min_area:
				continue

			objs.append((leftmost, rightmost, topmost, bottommost))

		return objs
