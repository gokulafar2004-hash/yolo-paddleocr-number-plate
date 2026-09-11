import cv2

IMAGE_PATH = r"D:\NumberPlateIdentifier\image\bike3.jpeg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError("Image could not be loaded")

print("Image loaded")
print("Image size:", image.shape)

# Approximate plate area for bike3.jpeg
plate = image[1000:1150, 1050:1200]

cv2.imwrite(
    r"D:\NumberPlateIdentifier\image\plate_crop.jpg",
    plate
)

print("Plate crop saved successfully")