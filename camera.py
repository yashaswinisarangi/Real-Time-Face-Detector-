import cv2

# Load the Haar cascade classifier
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class Video:
    def __init__(self):
        # Open webcam
        self.video = cv2.VideoCapture(0)  
        if not self.video.isOpened():
            raise RuntimeError("Error: Could not access the webcam")

    def __del__(self):
        # Release video when the object is deleted
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        # Read a frame from the video stream
        success, frame = self.video.read()
        if not success:
            print("Error: Could not read frame from camera")
            return None  # Prevent errors in Flask response

        # Convert frame to grayscale (face detection works better on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            x1, y1 = x + w, y + h
            
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 255), 2)

            # Add corner styling
            thickness = 6  # Line thickness for corners
            length = 30  # Corner length

            # Top-left corner
            cv2.line(frame, (x, y), (x + length, y), (255, 0, 255), thickness)
            cv2.line(frame, (x, y), (x, y + length), (255, 0, 255), thickness)

            # Top-right corner
            cv2.line(frame, (x1, y), (x1 - length, y), (255, 0, 255), thickness)
            cv2.line(frame, (x1, y), (x1, y + length), (255, 0, 255), thickness)

            # Bottom-left corner
            cv2.line(frame, (x, y1), (x + length, y1), (255, 0, 255), thickness)
            cv2.line(frame, (x, y1), (x, y1 - length), (255, 0, 255), thickness)

            # Bottom-right corner
            cv2.line(frame, (x1, y1), (x1 - length, y1), (255, 0, 255), thickness)
            cv2.line(frame, (x1, y1), (x1, y1 - length), (255, 0, 255), thickness)

        # Encode the frame as a JPEG image
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()
