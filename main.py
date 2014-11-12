import cv

font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8)

def writeText(text,x,y,size,frame):
	cv.PutText(frame,text, (x,y),font, size)


if __name__=='__main__':

	numframes = 0
	capture = cv.CaptureFromCAM(0)
	cv.NamedWindow('image')

	while True:
		frame = cv.QueryFrame(capture)
		writeText("Informacio:(esc per sortir)",50,50,200,frame)
		writeText("Frames: "+str(numframes),50,80,200,frame)
		cv.ShowImage('image',frame)

		k = cv.WaitKey(10)
		numframes += 1
		if k % 256 == 27:
			break
	cv.DestroyWindow('image')