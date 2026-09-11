import cv2
import easyocr
import re

IMAGE_PATH = r"D:\NumberPlateIdentifier\image\bike.jpeg"

# -----------------------------------------
# Load image
# -----------------------------------------
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not load image")
    exit()

# -----------------------------------------
# Crop number plate
# -----------------------------------------
plate = image[450:870, 100:820]

cv2.imwrite(
    r"D:\NumberPlateIdentifier\image\plate_crop.jpg",
    plate
)

# -----------------------------------------
# Resize
# -----------------------------------------
plate = cv2.resize(
    plate,
    None,
    fx=4,
    fy=4,
    interpolation=cv2.INTER_CUBIC
)

# -----------------------------------------
# Grayscale
# -----------------------------------------
gray = cv2.cvtColor(
    plate,
    cv2.COLOR_BGR2GRAY
)

# -----------------------------------------
# Denoising
# -----------------------------------------
gray = cv2.GaussianBlur(
    gray,
    (3, 3),
    0
)

# -----------------------------------------
# CLAHE contrast enhancement
# -----------------------------------------
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)

# -----------------------------------------
# Adaptive threshold
# -----------------------------------------
threshold = cv2.adaptiveThreshold(
    enhanced,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    11
)

cv2.imwrite(
    r"D:\NumberPlateIdentifier\image\plate_enhanced.jpg",
    threshold
)

# -----------------------------------------
# EasyOCR
# -----------------------------------------
reader = easyocr.Reader(
    ['en'],
    gpu=False
)

results = reader.readtext(
    threshold,
    detail=1,
    paragraph=False,
    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
)

print("\nOCR RESULTS")
print("=" * 50)

for result in results:

    text = result[1].upper()
    confidence = result[2]

    text = re.sub(
        r'[^A-Z0-9]',
        '',
        text
    )

    print(
        f"Text       : {text}"
    )

    print(
        f"Confidence : {confidence:.2f}"
    )

    print("-" * 50)
