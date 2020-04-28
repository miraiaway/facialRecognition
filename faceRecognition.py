import cv2

# Create cascade classifier object, take in an xml file that identifies the face
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Sets video source to default webcam
video_capture = cv2.VideoCapture(0)

while True: 
    # Capture frame by frame
    ret, frame = video_capture.read()

    # Convert the BRG image to GRAY why?
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()