from paddleocr import PaddleOCR
import os

IMAGE_PATH = r"D:\NumberPlateIdentifier\image\plate_enhanced.jpg"

if not os.path.exists(IMAGE_PATH):
    print("ERROR: Image not found")
    print(IMAGE_PATH)
    exit()

print("Loading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    device="cpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("PaddleOCR loaded successfully.")
print("Running OCR...")

results = ocr.predict(IMAGE_PATH)

print("\n" + "=" * 60)
print("PADDLEOCR RESULT")
print("=" * 60)

for result in results:
    print(result)

print("\nOCR TEST COMPLETE.")