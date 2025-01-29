import cv2
import numpy as np

net = cv2.dnn.readNet(r"C:\Users\ACER\Desktop\pyscript\object_classification_proj\yolov4.weights",
                      r"C:\Users\ACER\Desktop\pyscript\object_classification_proj\yolov4 (1).cfg")

classes = []
with open(r'C:\Users\ACER\Desktop\pyscript\object_classification_proj\coco.names', "r") as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
out_layer_indices = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in out_layer_indices]


img = cv2.imread('traffic_2.png')


resize_scale = 0.5  
img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)


height, width, _ = img.shape

# Define the trapezoid region of interest (ROI)
roi_pts = np.array([
    [int(width * 0.15), height],          # Bottom-left
    [int(width * 0.35), int(height * 0.38)],  # Top-left
    [int(width * 0.77), int(height * 0.38)],  # Top-right
    [int(width * 0.95), height]           # Bottom-right
], np.int32)

# Create a mask to keep only the region within the trapezoid
mask = np.zeros_like(img)
cv2.fillPoly(mask, [roi_pts], (255, 255, 255))

# Apply the mask to the image
masked_img = cv2.bitwise_and(img, mask)

# Define the positions for the lines at different meters
line_5m_y = int(height * 0.7)
line_10m_y = int(height * 0.5)
line_15m_y = int(height * 0.4)

# Draw blue lines within the trapezoid region
cv2.line(masked_img, (roi_pts[0][0], line_5m_y), (roi_pts[3][0], line_5m_y), (255, 0, 0), 2)  # 5 meters
cv2.line(masked_img, (roi_pts[1][0], line_10m_y), (roi_pts[2][0], line_10m_y), (255, 0, 0), 2)  # 10 meters
cv2.line(masked_img, (roi_pts[1][0], line_15m_y), (roi_pts[2][0], line_15m_y), (255, 0, 0), 2)  # 15 meters

# Prepare image for YOLO
blob = cv2.dnn.blobFromImage(masked_img, 0.00392, (700, 700), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Object detection
class_ids = []
confidences = []
boxes = []

# Dictionary to store vehicle information
vehicle_info_2 = {
    5: {},   # Vehicles within 5 meters
    10: {},  # Vehicles within 10 meters
    15: {}   # Vehicles within 15 meters
}

confidence_threshold = 0.3
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confidence_threshold:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Check if the object is within the trapezoid region
            if cv2.pointPolygonTest(roi_pts, (center_x, center_y), False) >= 0:
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

# Non-max suppression to remove overlapping bounding boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])

        # Identify in which meter line the vehicle is located
        if y + h // 2 > line_5m_y:
            meter = 5
        elif y + h // 2 > line_10m_y:
            meter = 10
        else:
            meter = 15

        # Store the information in the vehicle_info_2 dictionary
        if label not in vehicle_info_2[meter]:
            vehicle_info_2[meter][label] = 0
        vehicle_info_2[meter][label] += 1

        # Draw bounding box and label
        cv2.rectangle(masked_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(masked_img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

# Print vehicle information
print("Vehicle Info:", vehicle_info_2)

class VehicleInfo_8:
    def Vehfun_2():
        return vehicle_info_2

# Show the processed image
cv2.imshow("Detected Vehicles", masked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
