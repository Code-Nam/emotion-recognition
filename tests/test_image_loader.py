import os
import unittest
import tempfile

from src.image_loader import load_pgm, load_ppm

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
    

class TestPPMImageLoader(unittest.TestCase):

  def setUp(self):
    self.temp_file = tempfile.NamedTemporaryFile(suffix='.ppm', delete=False, mode='w')
    self.temp_file.write("P3\n4 2\n255\n")
    for _ in range(8):
      self.temp_file.write("10 20 30 ")
    self.temp_file.close()
    self.filename = self.temp_file.name

  def tearDown(self):
    if os.path.exists(self.filename):
      os.remove(self.filename)

  def test_load_ppm_dimensions(self):
    img = load_ppm(self.filename)
    self.assertEqual(len(img), 2)
    self.assertEqual(len(img[0]), 4)

  def test_load_ppm_pixel_values(self):
    img = load_ppm(self.filename)
    self.assertEqual(img[0][0], (10, 20, 30))
    self.assertEqual(img[1][3], (10, 20, 30))

  def test_load_ppm_invalid_format(self):
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False, mode='w') as f:
      f.write("P2\n4 2\n255\n")
      filename = f.name
    with self.assertRaises(ValueError):
      load_ppm(filename)
    os.remove(filename)

  def test_load_ppm_skips_comments(self):
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False, mode='w') as f:
      f.write("P3\n# comment\n2 2\n255\n0 0 0  255 255 255\n0 0 0  255 255 255\n")
      filename = f.name
    img = load_ppm(filename)
    self.assertEqual(img[0][0], (0, 0, 0))
    self.assertEqual(img[0][1], (255, 255, 255))
    os.remove(filename)


if __name__ == '__main__':
  unittest.main()
