import re
import cv2
from paddleocr import PaddleOCR

# =========================================================
# CONFIG
# =========================================================

IMAGE_PATH = r"D:\NumberPlateIdentifier\image\bullet.jpeg"
OUTPUT_DIR = r"D:\NumberPlateIdentifier"
SAVE_DEBUG_IMAGES = False          # set True if you want to inspect preprocessing
MIN_TEXT_LEN = 2
MIN_CONFIDENCE = 0.50

UPPER_PLATE_RE = re.compile(r"[A-Z]{2}\d{2}[A-Z]{1,3}")
NUMBER_PLATE_RE = re.compile(r"\d{4}")
NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")

print("Loading PaddleOCR...")
ocr = PaddleOCR(lang="en")


# =========================================================
# HELPERS
# =========================================================

def clean_text(text: str) -> str:
    """Uppercase and strip everything except A-Z0-9."""
    return NON_ALNUM_RE.sub("", text.upper())


def run_ocr(image, name):
    """Run PaddleOCR on a single image and return [(clean_text, score), ...]."""
    print(f"\nRunning OCR: {name}")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    result = ocr.predict(image)

    found = []
    for res in result:
        data = res.json
        if "res" in data:
            data = data["res"]

        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])

        for text, score in zip(texts, scores):
            print(f"Text: {text}  Confidence: {score:.2f}")
            cleaned = clean_text(text)
            if len(cleaned) >= MIN_TEXT_LEN:
                found.append((cleaned, float(score)))

    return found


def extract_plate(results):
    """
    Given a list of (clean_text, score) tuples, look for an upper part
    (state/region + series, e.g. TN75AC) and a number part (e.g. 3490).
    Returns (upper_part, number_part), either may be None.
    """
    upper_part = None
    number_part = None

    for text, score in results:
        if score < MIN_CONFIDENCE:
            continue
        if upper_part is None and UPPER_PLATE_RE.fullmatch(text):
            upper_part = text
        elif number_part is None and NUMBER_PLATE_RE.fullmatch(text):
            number_part = text

    return upper_part, number_part


def preprocess(image):
    """Resize, grayscale, CLAHE-enhance and sharpen. Returns the sharpened
    grayscale image plus the intermediate stages (for optional debug save)."""
    resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced_gray, (0, 0), 3)
    sharpened_gray = cv2.addWeighted(enhanced_gray, 1.8, blur, -0.8, 0)

    return resized, gray, sharpened_gray


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(IMAGE_PATH)
if image is None:
    print("ERROR: Image could not be loaded.")
    print(IMAGE_PATH)
    raise SystemExit(1)

print("Image loaded successfully")
print("Original size:", image.shape)

resized, gray, sharpened_gray = preprocess(image)

if SAVE_DEBUG_IMAGES:
    cv2.imwrite(f"{OUTPUT_DIR}\\plate_gray.jpg", gray)
    cv2.imwrite(f"{OUTPUT_DIR}\\plate_enhanced.jpg", sharpened_gray)
    print("\nPreprocessed images saved.")

sharpened = cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2BGR)


# =========================================================
# PASS 1: single best preprocessed image
# =========================================================

results = run_ocr(sharpened, "Sharpened (primary pass)")
upper_part, number_part = extract_plate(results)


# =========================================================
# PASS 2 (fallback): split into upper/lower lines
# Only bother if pass 1 didn't already find both parts.
# =========================================================

if not (upper_part and number_part):
    height = sharpened_gray.shape[0]

    upper_crop = cv2.cvtColor(
        sharpened_gray[: int(height * 0.55), :], cv2.COLOR_GRAY2BGR
    )
    lower_crop = cv2.cvtColor(
        sharpened_gray[int(height * 0.45):, :], cv2.COLOR_GRAY2BGR
    )

    print("\n========================================")
    print("FALLBACK: UPPER/LOWER LINE OCR")
    print("========================================")

    line_results = run_ocr(upper_crop, "Upper Line")
    line_results += run_ocr(lower_crop, "Lower Line")

    results += line_results
    if not upper_part or not number_part:
        u2, n2 = extract_plate(line_results)
        upper_part = upper_part or u2
        number_part = number_part or n2


# =========================================================
# CLEAN / FINAL RESULTS
# =========================================================

print("\n========================================")
print("ALL VALID OCR RESULTS")
print("========================================")
for text, score in results:
    print(f"{text:15} Confidence: {score:.2f}")

print("\n========================================")
print("LICENSE PLATE RESULT")
print("========================================")

if upper_part and number_part:
    final_plate = upper_part + number_part
    print("State/Region :", upper_part[:2])
    print("Plate part   :", upper_part)
    print("Number       :", number_part)
    print()
    print("FINAL PLATE  :", final_plate)
elif upper_part:
    print("Detected plate:", upper_part)
else:
    print("No valid plate detected.")

print("========================================")
print("TEST COMPLETED")
print("========================================")