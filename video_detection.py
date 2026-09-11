from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11m.pt")

# Detect objects in the video
results = model.predict(
    source=r"D:\NumberPlateIdentifier\sample 1.mp4",
    show=True,
    conf=0.4
)

print("Video detection completed")