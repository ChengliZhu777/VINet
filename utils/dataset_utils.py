import six

from PIL import Image


def lmdb_image_buffer_parse(image_buf):
    buf = six.BytesIO()
    buf.write(image_buf)
    buf.seek(0)

    try:
        image = Image.open(buf).convert('RGB')
        return True, image
    except IOError:
        return False, None


def fullwidth_to_halfwidth(string):
    half_string = ''
    for c in string:
        inside_code = ord(c)
      
