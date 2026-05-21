"""
test_generate_faces.py
Unit tests for the generate_faces module.
"""

import unittest
import random
import os
import tempfile
import math
import shutil
from pathlib import Path

# Import the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from generate_faces import (
    clamp, make_canvas, fill_circle, draw_circle, draw_arc, draw_line,
    draw_eyebrow, noise_gaussian,
    random_params, face_smiling, face_neutral, face_sad, face_angry, face_surprised,
    random_params_color, face_smiling_color, face_neutral_color,
    face_sad_color, face_angry_color, face_surprised_color,
    save_pgm, save_ppm, GENERATORS, COLOR_GENERATORS,
    split_counts, generate_dataset,
    WIDTH, HEIGHT
)
from image_loader import load_pgm, load_ppm


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_clamp_within_bounds(self):
        """Test clamping values within bounds."""
        self.assertEqual(clamp(128), 128)
        self.assertEqual(clamp(0), 0)
        self.assertEqual(clamp(255), 255)

    def test_clamp_below_minimum(self):
        """Test clamping below minimum."""
        self.assertEqual(clamp(-10), 0)
        self.assertEqual(clamp(-100), 0)

    def test_clamp_above_maximum(self):
        """Test clamping above maximum."""
        self.assertEqual(clamp(256), 255)
        self.assertEqual(clamp(1000), 255)

    def test_clamp_custom_bounds(self):
        """Test clamping with custom bounds."""
        self.assertEqual(clamp(5, lo=10, hi=20), 10)
        self.assertEqual(clamp(15, lo=10, hi=20), 15)
        self.assertEqual(clamp(25, lo=10, hi=20), 20)

    def test_clamp_float_rounding(self):
        """Test that clamping rounds floats."""
        self.assertEqual(clamp(128.4), 128)
        self.assertEqual(clamp(128.6), 129)


class TestCanvasCreation(unittest.TestCase):
    """Test canvas creation."""

    def test_make_canvas_default(self):
        """Test creating default canvas."""
        canvas = make_canvas()
        self.assertEqual(len(canvas), HEIGHT)
        self.assertEqual(len(canvas[0]), WIDTH)
        # Default background should be around 215
        self.assertGreaterEqual(canvas[0][0], 200)
        self.assertLessEqual(canvas[0][0], 230)

    def test_make_canvas_custom_bg(self):
        """Test creating canvas with custom background."""
        canvas = make_canvas(bg=100)
        self.assertTrue(all(pixel == 100 for row in canvas for pixel in row))

    def test_make_canvas_with_rng(self):
        """Test creating canvas with randomized background."""
        rng = random.Random(42)
        canvas = make_canvas(rng=rng)
        bg = canvas[0][0]
        self.assertGreaterEqual(bg, 200)
        self.assertLessEqual(bg, 230)


class TestRandomParams(unittest.TestCase):
    """Test random parameter generation."""

    def test_random_params_structure(self):
        """Test that random_params returns valid parameter dict."""
        rng = random.Random(42)
        params = random_params(8, 8, rng)

        required_keys = {'cx', 'cy', 'scale', 'dark', 'bg', 'rng'}
        self.assertEqual(set(params.keys()), required_keys)

    def test_random_params_scale(self):
        """Test that scale is within expected bounds."""
        rng = random.Random(42)
        for _ in range(10):
            params = random_params(8, 8, rng)
            self.assertGreaterEqual(params['scale'], 0.9)
            self.assertLessEqual(params['scale'], 1.1)

    def test_random_params_dark(self):
        """Test that dark value is within expected bounds."""
        rng = random.Random(42)
        for _ in range(10):
            params = random_params(8, 8, rng)
            self.assertGreaterEqual(params['dark'], 25)
            self.assertLessEqual(params['dark'], 55)

    def test_random_params_bg(self):
        """Test that background value is within expected bounds."""
        rng = random.Random(42)
        for _ in range(10):
            params = random_params(8, 8, rng)
            self.assertGreaterEqual(params['bg'], 200)
            self.assertLessEqual(params['bg'], 230)


class TestDrawingPrimitives(unittest.TestCase):
    """Test drawing primitive functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.canvas = make_canvas(bg=255)
        self.rng = random.Random(42)

    def test_fill_circle_modifies_canvas(self):
        """Test that fill_circle modifies the canvas."""
        original = [row[:] for row in self.canvas]
        fill_circle(self.canvas, 8, 8, 2, 100)
        # Check that center was modified
        self.assertNotEqual(self.canvas[8][8], original[8][8])

    def test_fill_circle_boundary(self):
        """Test that fill_circle respects canvas boundaries."""
        # This should not raise an error even near edges
        fill_circle(self.canvas, 1, 1, 10, 100)
        fill_circle(self.canvas, WIDTH-1, HEIGHT-1, 10, 100)
        # Canvas should still have valid dimensions
        self.assertEqual(len(self.canvas), HEIGHT)
        self.assertEqual(len(self.canvas[0]), WIDTH)

    def test_draw_circle_creates_outline(self):
        """Test that draw_circle creates an outline."""
        canvas = make_canvas(bg=255)
        draw_circle(canvas, 8, 8, 3, 100)
        # Should have some pixels modified
        self.assertTrue(any(pixel != 255 for row in canvas for pixel in row))

    def test_draw_line_modifies_canvas(self):
        """Test that draw_line modifies the canvas."""
        canvas = make_canvas(bg=255)
        draw_line(canvas, 2, 2, 14, 14, 100)
        # Should have some pixels modified
        self.assertTrue(any(pixel != 255 for row in canvas for pixel in row))

    def test_draw_arc_modifies_canvas(self):
        """Test that draw_arc modifies the canvas."""
        canvas = make_canvas(bg=255)
        draw_arc(canvas, 8, 8, 5, 0, math.pi, 100)
        # Should have some pixels modified
        self.assertTrue(any(pixel != 255 for row in canvas for pixel in row))

    def test_draw_eyebrow_modifies_canvas(self):
        """Test that draw_eyebrow modifies the canvas."""
        canvas = make_canvas(bg=255)
        draw_eyebrow(canvas, 8, 8, 3, 1, 100, 1)
        # Should have some pixels modified
        self.assertTrue(any(pixel != 255 for row in canvas for pixel in row))


class TestNoiseFunction(unittest.TestCase):
    """Test noise functions."""

    def test_noise_gaussian_modifies_canvas(self):
        """Test that Gaussian noise modifies the canvas."""
        canvas = make_canvas(bg=128)
        original = [row[:] for row in canvas]
        rng = random.Random(42)
        noise_gaussian(canvas, rng, sigma=20)
        # At least some pixels should be different
        differences = sum(1 for i in range(HEIGHT) for j in range(WIDTH)
                         if canvas[i][j] != original[i][j])
        self.assertGreater(differences, 0)

    def test_noise_gaussian_clamps_values(self):
        """Test that Gaussian noise clamps values to valid range."""
        canvas = make_canvas(bg=128)
        rng = random.Random(42)
        noise_gaussian(canvas, rng, sigma=50)
        # All values should be in valid range
        for row in canvas:
            for pixel in row:
                self.assertGreaterEqual(pixel, 0)
                self.assertLessEqual(pixel, 255)


class TestFaceGenerators(unittest.TestCase):
    """Test face generator functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.rng = random.Random(42)

    def test_face_smiling_returns_canvas(self):
        """Test that face_smiling returns a valid canvas."""
        canvas = face_smiling(self.rng)
        self.assertEqual(len(canvas), HEIGHT)
        self.assertEqual(len(canvas[0]), WIDTH)

    def test_face_neutral_returns_canvas(self):
        """Test that face_neutral returns a valid canvas."""
        canvas = face_neutral(self.rng)
        self.assertEqual(len(canvas), HEIGHT)
        self.assertEqual(len(canvas[0]), WIDTH)

    def test_face_sad_returns_canvas(self):
        """Test that face_sad returns a valid canvas."""
        canvas = face_sad(self.rng)
        self.assertEqual(len(canvas), HEIGHT)
        self.assertEqual(len(canvas[0]), WIDTH)

    def test_face_angry_returns_canvas(self):
        """Test that face_angry returns a valid canvas."""
        canvas = face_angry(self.rng)
        self.assertEqual(len(canvas), HEIGHT)
        self.assertEqual(len(canvas[0]), WIDTH)

    def test_face_surprised_returns_canvas(self):
        """Test that face_surprised returns a valid canvas."""
        canvas = face_surprised(self.rng)
        self.assertEqual(len(canvas), HEIGHT)
        self.assertEqual(len(canvas[0]), WIDTH)

    def test_all_generators_produce_valid_pixels(self):
        """Test that all generators produce valid pixel values."""
        for label, gen_fn in GENERATORS.items():
            rng = random.Random(42)
            canvas = gen_fn(rng)
            for row in canvas:
                for pixel in row:
                    self.assertGreaterEqual(pixel, 0,
                                          f"{label} produced pixel < 0")
                    self.assertLessEqual(pixel, 255,
                                        f"{label} produced pixel > 255")

    def test_generators_reproducible_with_same_seed(self):
        """Test that generators produce same output with same seed."""
        rng1 = random.Random(42)
        canvas1 = face_smiling(rng1)

        rng2 = random.Random(42)
        canvas2 = face_smiling(rng2)

        self.assertEqual(canvas1, canvas2)

    def test_generators_different_with_different_seed(self):
        """Test that generators produce different output with different seed."""
        rng1 = random.Random(42)
        canvas1 = face_smiling(rng1)

        rng2 = random.Random(43)
        canvas2 = face_smiling(rng2)

        self.assertNotEqual(canvas1, canvas2)


class TestPGMSerialization(unittest.TestCase):
    """Test PGM file saving and loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.pgm')

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)

    def test_save_pgm_creates_file(self):
        """Test that save_pgm creates a file."""
        canvas = make_canvas(bg=128)
        save_pgm(canvas, self.test_file)
        self.assertTrue(os.path.exists(self.test_file))

    def test_save_pgm_file_content(self):
        """Test that save_pgm writes correct PGM format."""
        canvas = make_canvas(bg=128)
        save_pgm(canvas, self.test_file)

        with open(self.test_file, 'r') as f:
            lines = f.readlines()

        # Check PGM header
        self.assertEqual(lines[0].strip(), 'P2')
        self.assertIn('test.pgm', lines[1])  # Comment
        self.assertEqual(lines[2].strip(), f'{WIDTH} {HEIGHT}')
        self.assertEqual(lines[3].strip(), '255')


class TestDatasetSplits(unittest.TestCase):
    """Test train/validation/test dataset splitting."""

    def test_split_counts(self):
        """Test that split counts follow a 60/20/20 pattern."""
        self.assertEqual(split_counts(10), (6, 2, 2))
        self.assertEqual(split_counts(5), (3, 1, 1))
        self.assertEqual(split_counts(1), (0, 0, 1))

    def test_generate_dataset_creates_split_directories(self):
        """Test that generate_dataset writes files into split directories."""
        temp_dir = tempfile.mkdtemp()
        try:
            generate_dataset(temp_dir, 5, 42)

            expected_counts = {
                "train": 3,
                "validation": 1,
                "test": 1,
            }

            for split_name, expected_count in expected_counts.items():
                for label in GENERATORS:
                    class_dir = os.path.join(temp_dir, split_name, label)
                    self.assertTrue(os.path.isdir(class_dir))
                    files = [name for name in os.listdir(class_dir) if name.endswith('.pgm')]
                    self.assertEqual(len(files), expected_count)
        finally:
            shutil.rmtree(temp_dir)


class TestColorFaceGenerators(unittest.TestCase):

    def setUp(self):
        self.rng = random.Random(42)

    def test_all_color_generators_produce_valid_pixels(self):
        for label, gen_fn in COLOR_GENERATORS.items():
            rng = random.Random(42)
            canvas = gen_fn(rng)
            self.assertEqual(len(canvas), HEIGHT)
            self.assertEqual(len(canvas[0]), WIDTH)
            for row in canvas:
                for pixel in row:
                    self.assertIsInstance(pixel, tuple, f"{label} pixel is not a tuple")
                    self.assertTrue(all(0 <= c <= 255 for c in pixel),
                                    f"{label} has channel out of [0,255]")

    def test_color_generators_reproducible(self):
        rng1, rng2 = random.Random(42), random.Random(42)
        self.assertEqual(face_smiling_color(rng1), face_smiling_color(rng2))

    def test_color_generators_differ_from_grayscale(self):
        rng1, rng2 = random.Random(42), random.Random(42)
        self.assertNotEqual(face_smiling(rng1), face_smiling_color(rng2))


class TestPPMSerialization(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.ppm')

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)

    def test_save_ppm_creates_file(self):
        canvas = face_smiling_color(random.Random(42))
        save_ppm(canvas, self.test_file)
        self.assertTrue(os.path.exists(self.test_file))

    def test_save_ppm_header(self):
        canvas = face_smiling_color(random.Random(42))
        save_ppm(canvas, self.test_file)
        with open(self.test_file, 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines[0].strip(), 'P3')
        self.assertIn('test.ppm', lines[1])
        self.assertEqual(lines[2].strip(), f'{WIDTH} {HEIGHT}')
        self.assertEqual(lines[3].strip(), '255')

    def test_save_and_reload_roundtrip(self):
        canvas = face_smiling_color(random.Random(42))
        save_ppm(canvas, self.test_file)
        self.assertEqual(canvas, load_ppm(self.test_file))


class TestColorDatasetGeneration(unittest.TestCase):

    def test_generate_dataset_color_creates_ppm_files(self):
        temp_dir = tempfile.mkdtemp()
        try:
            generate_dataset(temp_dir, 5, 42, color=True)
            for split_name, expected in [("train", 3), ("validation", 1), ("test", 1)]:
                for label in COLOR_GENERATORS:
                    class_dir = os.path.join(temp_dir, split_name, label)
                    self.assertTrue(os.path.isdir(class_dir))
                    files = [n for n in os.listdir(class_dir) if n.endswith('.ppm')]
                    self.assertEqual(len(files), expected)
        finally:
            shutil.rmtree(temp_dir)


class TestGeneratorsMap(unittest.TestCase):
    """Test the GENERATORS mapping."""

    def test_all_expressions_in_generators(self):
        """Test that all expected expressions are in GENERATORS."""
        expected = {'smiling', 'neutral', 'sad', 'angry', 'surprised'}
        self.assertEqual(set(GENERATORS.keys()), expected)

    def test_generators_are_callable(self):
        """Test that all generators are callable."""
        for label, gen_fn in GENERATORS.items():
            self.assertTrue(callable(gen_fn),
                          f"GENERATORS['{label}'] is not callable")


if __name__ == '__main__':
    unittest.main()
