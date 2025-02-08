import cv2

# Open the camera
video = cv2.VideoCapture(0) 


faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
# Read a frame from the video feed
    ret, frame = video.read()  
    faces = faceDetect.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:

        x1,y1=x+w, y+h
# cv2.shape(frame, pt1, pt2, color, thickness)
        cv2.rectangle(frame, (x,y), (x+w, y+h),(255,0,255), 1)
# top left
        cv2.line(frame,(x,y), (x+30,y), (255,0,255), 6)
        cv2.line(frame, (x,y), (x, y+30), (255, 0 , 255), 6)
# top right
        cv2.line(frame,(x1,y), (x1-30,y), (255,0,255), 6)
        cv2.line(frame, (x1,y), (x1, y+30), (255, 0 , 255),6)
# bottom left
        cv2.line(frame,(x,y1), (x+30,y1), (255,0,255), 6)
        cv2.line(frame, (x,y1), (x, y1-30), (255, 0 , 255),6)
# bottom right
        cv2.line(frame,(x1,y1), (x1-30,y1), (255,0,255), 6)
        cv2.line(frame, (x1,y1), (x1, y1-30), (255, 0 , 255), 6)
        
    if not ret:
        print("Failed to capture video")
        break
# display frame
    cv2.imshow("Frame", frame)  
# Capture key press
    k = cv2.waitKey(1) & 0xFF

# Exit loop if 'q' is pressed
    if k == ord('q'):
        break

# Release the camera & close windows
video.release()
cv2.destroyAllWindows()
