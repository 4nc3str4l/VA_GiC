import numpy as np
import cv2

class Window:
	def __init__(self, w, h):
		self.__w = w
		self.__h = h

		self.__window = np.zeros((w, h, 3), 'double')
		self.__texts = []

	def open(self, title):
		self.__title = title
		cv2.namedWindow(title, cv2.WINDOW_NORMAL)

	def showAt(self, pos, im, axis = -1):
		pos = tuple(pos)
		if len(pos) != 4:
			print "Unexpected parameter pos"
			return False

		im = im.squeeze()
		if axis == -1:
			self.__window[pos[0]:pos[0]+pos[2],pos[1]:pos[1]+pos[3]] = im
		else:
			self.__window[pos[0]:pos[0]+pos[2],pos[1]:pos[1]+pos[3], axis] = im

		return True

	def showGrayAt(self, pos, im):
		pos = tuple(pos)
		if len(pos) != 4:
			print "Unexpected parameter pos"
			return False

		if len(im.shape) > 2 and im.shape[2] > 1:
			print "Only 1 channel images are allowed"
			return False

		im = im.squeeze()
		self.__window[pos[0]:pos[0]+pos[2],pos[1]:pos[1]+pos[3], 0] = im
		self.__window[pos[0]:pos[0]+pos[2],pos[1]:pos[1]+pos[3], 1] = im
		self.__window[pos[0]:pos[0]+pos[2],pos[1]:pos[1]+pos[3], 2] = im

		return True

	def buffer(self, size, datatype = 'double'):
		return np.zeros(size, datatype)

	def text(self, text, pos, size):
		self.__texts.append((text, pos, size))

	def render(self):
		renderMatrix = self.__window.copy()

		for t in self.__texts:
			cv2.putText(renderMatrix, t[0], t[1], cv2.FONT_HERSHEY_SIMPLEX, t[2], (255,255,255))
		self.__texts = []

		cv2.imshow(self.__title, renderMatrix)
