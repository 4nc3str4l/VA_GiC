import cv2

def writeText(text, pos, size, frame):
	cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (255,255,255))


if __name__=='__main__':

	numframes = 0
	capture = cv2.VideoCapture(0)
	cv2.namedWindow('image')

	while True:
		_,frame = capture.read()
		writeText("Informacio:(esc per sortir)", (50,50), 1, frame)
		writeText("Frames: " + str(numframes), (50,80), 1, frame)
		cv2.imshow('image', frame)

		k = cv2.waitKey(10)
		numframes += 1
		if k & 0xFF == 27:
			break

	cv2.destroyAllWindows()
	capture.release()
