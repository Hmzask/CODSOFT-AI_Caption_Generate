from PIL import Image, ImageOps

def load_image(img_file):
    img = Image.open(img_file).convert("RGB")
    img = ImageOps.exif_transpose(img)
    return img

