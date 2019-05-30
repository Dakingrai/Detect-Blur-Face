import numpy as np
import cv2
import matplotlib.pyplot as plt

#load cascade classifier training file for haarcascade
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def display_image(img):
	plt.imshow(convertToRGB(img))
	plt.show()


def mask_image(img,blurred_img,x,y,w,h):
	height, width, channels = img.shape
	mask = np.zeros((height, width, 3), dtype=np.uint8)
	mask = cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
	return np.where(mask!=np.array([255, 255, 255]), img, blurred_img)

def video(source=0):

	cap = cv2.VideoCapture(source)

	while(True):
	    # Capture frame-by-frame
		ret, frame = cap.read()

	    # Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
	    
	    #go over list of faces and draw them as rectangles on original colored img
		for (x, y, w, h) in faces:
			blurred_img = cv2.blur(frame,(100,100))
			out = mask_image(frame,blurred_img,x,y,w,h)

		try:
			cv2.imshow('frame', out)

		except UnboundLocalError:
			cv2.imshow('frame', gray)

		#print the number of faces found
		print('Faces found: ', len(faces))

		# cv2.imshow('frame', gray)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def picture(filename):
	img = cv2.imread(filename)

	#convert the test image to gray image as opencv face detector expects gray images
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#let's detect multiscale (some images may be closer to camera than others) images
	faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
	#go over list of faces and draw them as rectangles on original colored img
	for (x, y, w, h) in faces:
		blurred_img = cv2.blur(img,(100,100))
		out = mask_image(img,blurred_img,x, y, w, h)

	#print the number of faces found
	print('Faces found: ', len(faces))
	try:
		display_image(out)
	except UnboundLocalError:
		display_image(img)

if __name__ == '__main__':
	# empty argument for webcam as a source
	video('data/video.mp4')

	#picture('test.jpg')