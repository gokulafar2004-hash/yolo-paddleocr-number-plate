import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Load YOLO license plate model
model = YOLO("license-plate-finetune-v1m.pt")

# Load PaddleOCR
ocr = PaddleOCR(lang="en")

# Open video
video = cv2.VideoCapture("sample 1.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Detect license plates
    results = model(frame, conf=0.25)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop license plate
            plate = frame[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            # OCR
            ocr_result = ocr.predict(plate)

            # Draw detection box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display OCR result
            cv2.putText(
                frame,
                "Plate detected",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Number Plate Recognition", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
