from ultralytics import YOLO
import cv2
import os

# Load model once
model = YOLO("best.pt")

def predict(image_path):

    results = model(image_path, conf=0.25)[0]

    # draw predicted boxes
    plotted = results.plot()
    result_path = "static/results/output.jpg"
    cv2.imwrite(result_path, plotted)

    # ----- fracture decision logic (CLASS NAME BASED) -----
    status = "No Fracture Detected"
    color = "lime"

    boxes = results.boxes

    if boxes is not None and len(boxes) > 0:
        for cls in boxes.cls:
            class_name = results.names[int(cls)]

            if class_name.lower() == "fracture":
                status = "Fracture Detected"
                color = "red"
                break

    return result_path, status, color
