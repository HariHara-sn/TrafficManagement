import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet(r"C:\Users\ACER\Desktop\pyscript\object_classification_proj\yolov4.weights",
                      r"C:\Users\ACER\Desktop\pyscript\object_classification_proj\yolov4 (1).cfg ")

# Load classes
classes = []
with open(r'C:\Users\ACER\Desktop\pyscript\object_classification_proj\coco.names', "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layer names
layer_names = net.getLayerNames()
out_layer_indices = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in out_layer_indices]

# Load image
img = cv2.imread("img.png")

# Resize the image for processing
resize_scale = 0.3
img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

# Get image dimensions
height, width, _ = img.shape

# Define and draw the blue lines (for 5m, 15m, and 20m distances)
line_5m_y = int(height * 0.79)  # 5 meters
line_15m_y = int(height * 0.50)  # 15 meters
line_20m_y = int(height * 0.25)  # 20 meters

cv2.line(img, (0, line_5m_y), (int(width*0.6), line_5m_y), (255, 0, 0), 2)  # Blue line for 5 meters
cv2.line(img, (0, line_15m_y), (int(width*0.6), line_15m_y), (255, 0, 0), 2)  # Blue line for 15 meters
cv2.line(img, (0, line_20m_y), (int(width*0.7), line_20m_y), (255, 0, 0), 2)  # Blue line for 20 meters

# Dictionary to store the count and position of vehicles
vehicle_info = {
   
    5 : {},
    15: {},
    20: {}
}

# Prepare image for YOLO
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Object detection
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
        if label in ["car", "bicycle", "motorbike", "bus", "truck"]:
            # Draw bounding box and label
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
            # Determine the vehicle's distance from the line
            vehicle_position = ""
            if y + h > line_5m_y:
                vehicle_position = 5
            elif y + h > line_15m_y:
                vehicle_position = 15
            elif y + h > line_20m_y:
                vehicle_position = 20
            
            # Store the vehicle information in the dictionary
            if vehicle_position:
                if label in vehicle_info[vehicle_position]:
                    vehicle_info[vehicle_position][label] += 1
                else:
                    vehicle_info[vehicle_position][label] = 1

# Show the image with bounding boxes and blue lines
# cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the dictionary containing vehicle information
class VehicleInfo:
    def Vehfun():
        return vehicle_info


# print(vehicle_info)
