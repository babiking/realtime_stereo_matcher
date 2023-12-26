import re
import sys
import numpy as np


def read_pfm_file(pfm_file):
    """
    Read a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.
    """

    fp = open(pfm_file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = fp.readline().rstrip()
    if header.decode("ascii") == "PF":
        color = True
    elif header.decode("ascii") == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.search(r"(\d+)\s(\d+)", fp.readline().decode("ascii"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(fp.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    shape = (height, width, 3) if color else (height, width)
    data = np.flipud(np.fromfile(fp, endian + "f").reshape(shape))
    return data, scale


def write_pfm_file(pfm_file, image, scale=1):
    """
    Write a Numpy array to a PFM file.
    """
    fp = open(pfm_file, "wb")

    color = None

    if image.dtype.name != "float32":
        raise Exception("Image dtype must be float32.")

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif (
        len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
    ):  # greyscale
        color = False
    else:
        raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    fp.write(b"PF\n" if color else b"Pf\n")
    fp.write(b"%d %d\n" % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == "<" or endian == "=" and sys.byteorder == "little":
        scale = -scale

    fp.write(b"%f\n" % scale)

    image.tofile(fp)
