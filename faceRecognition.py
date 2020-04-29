import cv2

# Create cascade classifier object, take in an xml file that identifies the face
# This xml file contains information to detect the face
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Sets video source to default webcam
video_capture = cv2.VideoCapture(0)

while True: 
    # Capture frame by frame
    ret, frame = video_capture.read()

    # Convert the BRG image to GRAY 
    # Gray scale image is from 0-255 where RGB or BGR is three different layer images, color makes it more complex
    # Why use HSV? (unrelated to this post but keeping it for my notes)
    # HSV model describes the color to how the human eye preceives color
    # RGB is a combination  of primary colors
    # Hue = Represents the color
    # Saturation = Represents the amount of that respective color is mixed with white
    # Value = represents the amount of that respective color is mixed with black (gray level)
    # RGB cannot separate color information for luminance, HSV can be used to separate image luminance from color information 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectMultiScale -> Detects objects of different sizes in the input image, detected objects are returned as a list of rectangles
    #(image, scaleFactor, minNeighbors)
    # scale factor = specifies how much the image size is reduced at each image scale
    # It is a scale factor, your model has a fixed size defined during training (visible in xml), the size of face is detected in the image if present
    # 1.05 = reduce size by 5%, can increase the change fo a matching size with the model for detection, slower detection compared to 1.4 tho
    # minNeighbors = specifies how many neighbors each candidate rectangle should have to retain -> affects the quality of detected faces
    # Higher value reults in less detections but higher quality (3~6 is good)
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