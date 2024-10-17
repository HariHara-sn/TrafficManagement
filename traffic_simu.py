import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet(r"C:\Users\ACER\Desktop\pyscript\object_classification_proj\yolov4.weights",
                      r"C:\Users\ACER\Desktop\pyscript\object_classification_proj\yolov4 (1).cfg")
classes = []
with open('coco.names', "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
out_layer_indices = net.getUnconnectedOutLayers().flatten()  # Flatten to handle both cases
output_layers = [layer_names[i - 1] for i in out_layer_indices]  # Adjust index for Python's 0-based indexing

# Open video capture
cap = cv2.VideoCapture(r'D:\traffic_simulation\traffic_vedio.mp4')

# Set the scale for resizing (adjust as needed)
resize_scale = 0.3  # This reduces the frame size to 30%. Adjust this value for more or less zoom out.

while True:
    ret, img = cap.read()
    if not ret:
        break  # If no frame is captured, break the loop

    # Resize the image to zoom out
    img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    height, width, _ = img.shape

    # Define the positions of the blue lines
    line_5m_y = int(height * 0.79)  # 5 meters
    line_15m_y = int(height * 0.50)  # 15 meters
    line_20m_y = int(height * 0.25)   # 20 meters

    # Draw the blue lines indicating the distances
    cv2.line(img, (0, line_5m_y), (int(width*0.6), line_5m_y), (255, 0, 0), 2)  # Blue line for 5 meters
    cv2.line(img, (0, line_15m_y), (int(width*0.6), line_15m_y), (255, 0, 0), 2)  # Blue line for 15 meters
    cv2.line(img, (0, line_20m_y), (int(width*0.7), line_20m_y), (255, 0, 0), 2)  # Blue line for 20 meters

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to show on the screen (class id, object location, etc)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Check if the object is within the left side ROI
                if 0 <= center_x <= int(width * 0.7):
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Non-max suppression to remove overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label in ["car", "bicycle", "motorbike", "bus","truck"]:
                # Draw bounding box for the object
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Show the image
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()
