import cv2

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

    cv2.imshow("Traffic Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()