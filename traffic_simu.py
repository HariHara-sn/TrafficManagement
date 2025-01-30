import cv2
import numpy as np
import os

# Load YOLO model
net = cv2.dnn.readNet(r"C:\Users\ACER\Desktop\pyscript\object_classification_proj\yolov4.weights",
                      r"C:\Users\ACER\Desktop\pyscript\object_classification_proj\yolov4 (1).cfg")

# Load classes
classes = []
with open(r'C:\Users\ACER\Desktop\pyscript\object_classification_proj\coco.names', "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layer names
layer_names = net.getLayerNames()
out_layer_indices = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in out_layer_indices]

# Load image
img = cv2.imread('traffic_2.png')

# Resize the image for processing
resize_scale = 0.5
img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

# Get image dimensions
height, width, _ = img.shape

# Define the trapezoid region of interest (ROI)
roi_pts = np.array([
    [int(width * 0.15), height],  
    [int(width * 0.35), int(height * 0.38)],  
    [int(width * 0.77), int(height * 0.38)],  
    [int(width * 0.95), height]  
], np.int32)

# Create a mask and apply it to the image
mask = np.zeros_like(img)
cv2.fillPoly(mask, [roi_pts], (255, 255, 255))
masked_img = cv2.bitwise_and(img, mask)

# Draw lines at 5m, 10m, and 15m
line_5m_y = int(height * 0.7)
line_10m_y = int(height * 0.5)
line_15m_y = int(height * 0.4)

cv2.line(masked_img, (roi_pts[0][0], line_5m_y), (roi_pts[3][0], line_5m_y), (255, 0, 0), 2) 
cv2.line(masked_img, (roi_pts[1][0], line_10m_y), (roi_pts[2][0], line_10m_y), (255, 0, 0), 2)  
cv2.line(masked_img, (roi_pts[1][0], line_15m_y), (roi_pts[2][0], line_15m_y), (255, 0, 0), 2)  

# Prepare image for YOLO
blob = cv2.dnn.blobFromImage(masked_img, 0.00392, (700, 700), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Object detection
class_ids = []
confidences = []
boxes = []

vehicle_info_2 = {5: {}, 10: {}, 15: {}}
confidence_threshold = 0.3
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confidence_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            if cv2.pointPolygonTest(roi_pts, (center_x, center_y), False) >= 0:
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

# Non-max suppression to remove overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])

        if y + h // 2 > line_5m_y:
            meter = 5
        elif y + h // 2 > line_10m_y:
            meter = 10
        else:
            meter = 15

        if label not in vehicle_info_2[meter]:
            vehicle_info_2[meter][label] = 0
        vehicle_info_2[meter][label] += 1

        # Draw bounding box and label
        cv2.rectangle(masked_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(masked_img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

# Print detected vehicle info
print(vehicle_info_2)

# Ensure "newfolder" exists, if not, create it
output_folder = "VehicleCount"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the processed image in "newfolder"
output_path = os.path.join(output_folder, "detected_vehicles.png")
cv2.imwrite(output_path, masked_img)

print(f"Processed image saved at: {output_path}")
