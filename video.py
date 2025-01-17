import cv2


def detect_faces():
    # Load the pre-trained face detection cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing video from the webcam
    video_capture = cv2.VideoCapture(0)

    # Counter for generating unique filenames
    unique_faces = []

    while True:
        # Read each frame of the video
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
             # Check if the face is unique
            is_unique = True
            for face in unique_faces:
                (xf, yf, wf, hf) = face
                if x > xf and y > yf and x + w < xf + wf and y + h < yf + hf:
                    is_unique = False
                    break

            # If the face is unique, save it and add to the list of unique faces
            if is_unique:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                unique_faces.append((x, y, w, h))
                face_filename = f"face_{len(unique_faces)}.jpg"
                cv2.imwrite(face_filename, frame[y:y+h, x:x+w])




            # Save individual face images with unique filenames
            # face_img = frame[y:y+h, x:x+w]
            # face_filename = "face_" + str(face_counter) + ".jpg"
            # cv2.imwrite(face_filename, face_img)
            # face_counter += 1

        # Display the resulting frame with detected faces
        cv2.imshow('Video', frame)
        # Save the image of the last detected face
        # if len(faces) > 0:
        #     last_face = frame[y:y+h, x:x+w]
        #     cv2.imwrite('./test/33.jpg', last_face)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()


detect_faces()











import cv2

def detect_faces_and_save():
    # Load the pre-trained face detection cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing video from the webcam
    video_capture = cv2.VideoCapture(0)

    # List to store unique faces
    unique_faces = []

    while True:
        # Read each frame of the video
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Check if the face is unique
            is_unique = True
            for (xf, yf, wf, hf) in unique_faces:
                if (x >= xf and y >= yf and x + w <= xf + wf and y + h <= yf + hf) or (x <= xf and y <= yf and x + w >= xf + wf and y + h >= yf + hf):
                    is_unique = False
                    break

            # If the face is unique, save it and add to the list of unique faces
            if is_unique:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                unique_faces.append((x, y, w, h))
                face_filename = f"face_{len(unique_faces)}.jpg"
                cv2.imwrite(face_filename, frame[y:y+h, x:x+w])

        # Display the resulting frame with detected faces
        cv2.imshow('Video', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()


detect_faces_and_save()