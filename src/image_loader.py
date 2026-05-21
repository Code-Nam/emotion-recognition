import struct


def load_ppm(filename):
    """Load a P3 ASCII PPM file and return a 2-D array of (R, G, B) tuples."""
    with open(filename, "r") as f:
        line = f.readline().strip()
        if line != "P3":
            raise ValueError(f"Unsupported PPM format: {line}")
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PPM header")
            line = line.strip()
            if line.startswith("#") or len(line) == 0:
                continue
            break
        width, height = map(int, line.split())
        int(f.readline().strip())  # max_val
        values = [int(v) for v in f.read().split()]
    img = []
    for y in range(height):
        row = []
        for x in range(width):
            idx = (y * width + x) * 3
            row.append((values[idx], values[idx + 1], values[idx + 2]))
        img.append(row)
    return img


def load_pgm(filename):
    with open(filename, 'rb') as f:
        # Read the magic number
        magic_number = f.readline().strip()
        
        # Check if it's a supported PGM format (P5 for binary, P2 for ASCII)
        if magic_number not in [b'P5', b'P2']:
            raise ValueError("Unsupported PGM format: {}".format(magic_number))
        
        # Read next non-comment, non-empty line for width and height
        while True:
          line = f.readline()
          if not line:
            raise ValueError("Unexpected EOF while reading PGM header")
          line = line.strip()
          # Skip comment or empty lines
          if line.startswith(b'#') or len(line) == 0:
            continue
          break

        width, height = map(int, line.split())
        max_pixel_value = int(f.readline().strip())
        
        # Read pixel data
        if magic_number == b'P2':
          pixels = [int(v) for v in f.read().split()][:width * height]
        else:
          bytes_per_pixel = 1 if max_pixel_value < 256 else 2
          img_bytes = f.read(width * height * bytes_per_pixel)
          if bytes_per_pixel == 1:
            pixels = list(img_bytes)
          else:
            pixels = list(struct.unpack(f'>{width * height}H', img_bytes))
          
        img = [pixels[i * width:(i + 1) * width] for i in range(height)]
        
    return img

def downsample_2x(img):
    """Average-pool an image (grayscale or color) by a factor of 2."""
    h = len(img)
    w = len(img[0])
    out = []
    for y in range(0, h, 2):
        row = []
        for x in range(0, w, 2):
            pixels = [
                img[y + dy][x + dx]
                for dy in range(2) for dx in range(2)
                if y + dy < h and x + dx < w
            ]
            if isinstance(pixels[0], tuple):
                n = len(pixels)
                row.append(tuple(sum(p[c] for p in pixels) // n for c in range(3)))
            else:
                row.append(sum(pixels) // len(pixels))
        out.append(row)
    return out
