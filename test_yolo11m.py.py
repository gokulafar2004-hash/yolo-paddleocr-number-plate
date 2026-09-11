from ultralytics import YOLO

model = YOLO("yolo11m.pt")

results = model(r"D:\NumberPlateIdentifier\image\sample 1.mp3", show=True)

print("YOLO11m test completed")