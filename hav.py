import cv2
import numpy as np

# Load the YOLO model
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Set the input image dimensions
width, height = 640, 480

# Start the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    _, frame = video_capture.read()

    # Resize the frame to the input image dimensions
    frame = cv2.resize(frame, (width, height))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle and display "Human" text around the faces
        # Draw a rectangle and display "Human" text around the faces
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Human', (x, y-5), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
            # Detect objects in the frame using YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

            # Get the bounding boxes and class probabilities for the detected objects
        boxes, confidences, class_ids = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                    x, y = int(center_x - w/2), int(center_y - h/2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indices:
            #i = i[0]
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = str(class_ids[i])
                # Add text indicating that the detected object is not a human
            if label != '0':
                label = 'Non-Human'
                cv2.putText(frame, label, (x, y-5), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()