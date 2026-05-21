import os
import unittest
import tempfile

from src.image_loader import load_pgm, img_to_vectors

class TestPGMImageLoader(unittest.TestCase):
  def setUp(self):
    self.temp_file = tempfile.NamedTemporaryFile(suffix='.pgm',
                                                 delete=False,
                                                 mode='wb')
    
    # Write a simple 8x8 PGM image to the temporary file
    self.temp_file.write(b'P5\n')
    self.temp_file.write(b'8 8\n')
    self.temp_file.write(b'255\n')
    
    self.temp_file.write(bytes([0] * 64))  # Write 64 pixels (8x8) with value 0
    self.temp_file.close()
    self.filename = self.temp_file.name
    
  def tearDown(self):
    if os.path.exists(self.filename):
      os.remove(self.filename)
      
  def test_load_pgm_dimensions(self):
    img = load_pgm(self.filename)
    self.assertEqual(len(img), 8)  # Height should be 8
    self.assertEqual(len(img[0]), 8)  # Width should be 8
    
  def test_load_pgm_pixel_values(self):
    img = load_pgm(self.filename)
    self.assertEqual(img[0][0], 0)  # Top-left pixel should be 0
    self.assertEqual(img[7][7], 0)  # Bottom-right pixel should be 0
    
  def test_load_pgm_invalid_format(self):
    with tempfile.NamedTemporaryFile(suffix='.pgm', delete=False, mode='wb') as temp_file:
      temp_file.write(b'P3\n')  # Invalid magic number for PGM
      temp_file.write(b'8 8\n')
      temp_file.write(b'255\n')
      temp_file.write(bytes([0] * 64))
      filename = temp_file.name
    
    with self.assertRaises(ValueError):
      load_pgm(filename)
    
    os.remove(filename)
    
  def test_vectorize_8x8(self):
    img = load_pgm(self.filename)
    vectors = img_to_vectors(img, vector_size=8)
    
    self.assertEqual(len(vectors), 1)  # Should be one vector for an 8x8 image
    self.assertEqual(len(vectors[0]), 64)  # Each vector should have 64 values
    
  def test_vectorize_16x16(self):
    # Create a 16x16 PGM image
    with tempfile.NamedTemporaryFile(suffix='.pgm', delete=False, mode='wb') as temp_file:
      temp_file.write(b'P5\n')
      temp_file.write(b'16 16\n')
      temp_file.write(b'255\n')
      temp_file.write(bytes([0] * 256))  # Write 256 pixels (16x16) with value 0
      filename = temp_file.name
    
    img = load_pgm(filename)
    vectors = img_to_vectors(img, vector_size=8)
    
    self.assertEqual(len(vectors), 4)  # Should be four vectors for a 16x16 image
    self.assertEqual(len(vectors[0]), 64)  # Each vector should have 64 values
    
    os.remove(filename)
    
if __name__ == '__main__':
  unittest.main()
