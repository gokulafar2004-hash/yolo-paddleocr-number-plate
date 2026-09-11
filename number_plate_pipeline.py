import cv2
import re
import numpy as np
from paddleocr import PaddleOCR


# ============================================================
# 1. LOAD IMAGE
# ============================================================

IMAGE_PATH = r"D:\NumberPlateIdentifier\image\bike3.jpeg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

print("Image loaded successfully")
print("Original size:", image.shape)


# ============================================================
# 2. RESIZE
# ============================================================

scale = 3

resized = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)

cv2.imwrite(
    r"D:\NumberPlateIdentifier\image\plate_resized.jpg",
    resized
)


# ============================================================
# 3. GRAYSCALE
# ============================================================

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

cv2.imwrite(
    r"D:\NumberPlateIdentifier\image\plate_gray.jpg",
    gray
)


# ============================================================
# 4. CONTRAST ENHANCEMENT
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)

cv2.imwrite(
    r"D:\NumberPlateIdentifier\image\plate_contrast.jpg",
    enhanced
)


# ============================================================
# 5. SHARPENING
# ============================================================

kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

sharpened = cv2.filter2D(
    enhanced,
    -1,
    kernel
)

cv2.imwrite(
    r"D:\NumberPlateIdentifier\image\plate_sharp.jpg",
    sharpened
)


# ============================================================
# 6. PADDLEOCR
# ============================================================

print("\nLoading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("PaddleOCR loaded successfully")


# ============================================================
# 7. RUN OCR
# ============================================================

print("\nRunning OCR...")

ocr_input = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

result = ocr.predict(ocr_input)


# ============================================================
# 8. EXTRACT TEXT
# ============================================================

all_text = []

for res in result:

    if hasattr(res, "json"):
        data = res.json
    else:
        data = res

    if isinstance(data, str):
        try:
            data = data
        except:
            continue

    if isinstance(data, dict):

        rec_texts = data.get("rec_texts", [])

        for text in rec_texts:

            if text:
                all_text.append(str(text))


print("\nRaw OCR output:")
print(all_text)


# ============================================================
# 9. CLEAN TEXT
# ============================================================

combined_text = "".join(all_text)

# Convert lowercase to uppercase
combined_text = combined_text.upper()

# Keep ONLY A-Z and 0-9
cleaned_text = re.sub(
    r"[^A-Z0-9]",
    "",
    combined_text
)


# ============================================================
# 10. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 50)
print("LICENSE PLATE RESULT")
print("=" * 50)

if cleaned_text:

    print("Detected plate:", cleaned_text)

else:

    print("No valid plate detected")

print("=" * 50)