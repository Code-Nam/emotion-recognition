import struct

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
        bytes_per_pixel = 1 if max_pixel_value < 256 else 2
        img_bytes = f.read(width * height * bytes_per_pixel)
        
        # Convert bytes to a 2D list of pixel values
        if bytes_per_pixel == 1:
          pixels = list(img_bytes)
        else:
          pixels = list(struct.unpack(f'>{width * height}H', img_bytes))
          
        img = [pixels[i * width:(i + 1) * width] for i in range(height)]
        
    return img
  
# Otsu method not implemented here, but we can use a simple threshold for normalization
def normalize_to_binary(img, threshold=128):
    binary_img = []
    for row in img:
        binary_row = [1 if pixel >= threshold else 0 for pixel in row]
        binary_img.append(binary_row)
    return binary_img
  
def img_to_vectors(img, vector_size=8):
    height = len(img)
    width = len(img[0])
    vectors = []
    
    for y in range(0, height, vector_size):
        for x in range(0, width, vector_size):
            vector = []
            for dy in range(vector_size):
              if y + dy < height:
                for dx in range(vector_size):
                  if x + dx < width:
                    vector.append(img[y + dy][x + dx])
                  else:
                    vector.append(0)  # Pad with zeros if out of bounds

              else:
                vector.extend([0] * vector_size)  # Pad with zeros if out of bounds

            vectors.append(vector)
    
    return vectors