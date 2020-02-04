import argparse
import numpy as np
from PIL import Image
import glob
from w2rgb import w2rgb
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil


parser = argparse.ArgumentParser(description="Make rainbows")
parser.add_argument('-d', '--dynamic', help="Add a dynamic range multiplier",
                    type=int, default=[], nargs="+")
parser.add_argument('-e', '--extension', help="Extension to look for")
args = parser.parse_args()


def load_images(extension=".JPG"):
    """Loads all images from src directory and returns a list of numpy arrays
    """

    image_list = []
    for i, filename in enumerate(sorted(glob.glob(f"src/*{extension}"))):
        if i == 0:
            print(f"Reading images of form: {filename}")
        enhanced_im = Image.open(filename)
        converted = enhanced_im.convert("L")
        image_list.append(np.array(converted))

    print("Finished reading images")

    return image_list


def save_image(image, filename="output.bmp"):
    """Saves the specified image at filename
    """

    image.save(filename)
    return


def balance_image(image):
    im = Image.fromarray(image)
    corrected = to_pil(cca.stretch(from_pil(im)))
    return corrected


def assign_wavelengths(n):
    """Returns list of wavelengths correlating to n images
    """

    min = 380
    max = 780

    return [(min + ((max - min)/n) * x) for x in range(1, n + 1)]


def tint(image, wavelength):
    """Returns a tinted version of the image using the wavelength
    """

    alphas = np.zeros(image.shape, dtype=np.float32)
    image = np.stack((image, image, image, alphas), axis=-1)
    new = np.zeros(image.shape)
    color = new.astype(np.float32) / 255
    color[:] = [*w2rgb(wavelength)] + [1.0]
    gray = image.astype(np.float32) / 255

    multiplied = gray * color

    return (multiplied / np.max(multiplied) * 255).astype(np.uint8)


def combine(images, L=1):
    """Combines each of the numpy array images using the formula
    log(L * sum(exp(pixel values)))
    """
    temp = []

    for image in images:
        gray = image.astype(np.float32) / 255
        temp.append(gray)

    output = np.log2((1/len(temp)) * sum(map(lambda x: 2 ** (L * x), temp)))

    return ((output / np.max(output)) * 255).astype(np.uint8)


def main():
    images = load_images(args.extension)
    tinted = []
    for image, w in zip(images, assign_wavelengths(len(images))):
        tinted.append(tint(image, w))
    for i, pic in enumerate(tinted):
        save_image(Image.fromarray(pic), f"tinted/{i}_tinted.bmp")
    if len(args.dynamic) == 0:
        final = combine(tinted)
        save_image(balance_image(final))
    elif len(args.dynamic) == 1:
        final = combine(tinted, args.dynamic)
        save_image(balance_image(final))
    else:
        for d in args.dynamic:
            final = combine(tinted, d)
            save_image(balance_image(final), f"output_{d}.bmp")


if __name__ == "__main__":
    main()
