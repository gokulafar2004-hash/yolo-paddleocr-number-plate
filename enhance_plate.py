import cv2
import numpy as np

INPUT = r"D:\NumberPlateIdentifier\image\plate_crop.jpg"
OUTPUT = r"D:\NumberPlateIdentifier\image\plate_enhanced.jpg"

# Load cropped plate
image = cv2.imread(INPUT)

if image is None:
    raise FileNotFoundError("plate_crop.jpg not found")

print("Plate crop loaded")
print("Original crop size:", image.shape)

# ------------------------------------------------------------
# 1. ENLARGE PLATE
# ------------------------------------------------------------

image = cv2.resize(
    image,
    None,
    fx=4,
    fy=4,
    interpolation=cv2.INTER_CUBIC
)

print("Enlarged size:", image.shape)


# ------------------------------------------------------------
# 2. GRAYSCALE
# ------------------------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ------------------------------------------------------------
# 3. CONTRAST ENHANCEMENT
# ------------------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)


# ------------------------------------------------------------
# 4. DENOISING
# ------------------------------------------------------------

denoised = cv2.fastNlMeansDenoising(
    enhanced,
    None,
    10,
    7,
    21
)


# ------------------------------------------------------------
# 5. SHARPENING
# ------------------------------------------------------------

kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

sharpened = cv2.filter2D(
    denoised,
    -1,
    kernel
)


# ------------------------------------------------------------
# 6. SAVE
# ------------------------------------------------------------

cv2.imwrite(
    OUTPUT,
    sharpened
)

print("Enhanced plate saved successfully:")
print(OUTPUT)