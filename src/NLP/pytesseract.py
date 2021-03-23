from PIL import Image

from pytesseract import image_to_string

from config import CONFIG

"""Need to install tesseract"""

if __name__ == "__main__":
    filename = str(CONFIG.notebook.parent / "enhanceIT.png")
    text = image_to_string(Image.open(filename))
    print(text)
