from ultralytics import YOLO
import cv2

# Load license plate model
model = YOLO("license-plate-finetune-v1m.pt")

# Open video
cap = cv2.VideoCapture(r"D:\NumberPlateIdentifier\sample 1.mp4")

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished")
        break

    # Detect number plates
    results = model(frame, conf=0.4)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("Number Plate Detection", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()