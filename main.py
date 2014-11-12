import cv

if __name__=='__main__':
	capture = cv.CaptureFromCAM(0)

	cv.NamedWindow('image')

	while True:
		frame = cv.QueryFrame(capture)
		cv.ShowImage('image',frame)

		k = cv.WaitKey(10)

		if k % 256 == 27:
			break
	cv.DestroyWindow('image')