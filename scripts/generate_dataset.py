#!/usr/bin/env python3
"""Generates a synthetic grayscale image dataset for the GPU pipeline.

Images are deliberately low-contrast (values squeezed into a narrow band) and
contain random shapes plus noise, so that histogram equalization and edge
detection both produce clearly visible results.

Usage:
  python3 scripts/generate_dataset.py --output_dir data --num_images 1000
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw


def make_image(rng, width, height):
  low = int(rng.integers(20, 100))
  high = low + int(rng.integers(50, 90))
  ramp_x = np.linspace(0.0, 1.0, width)[None, :]
  ramp_y = np.linspace(0.0, 1.0, height)[:, None]
  angle = rng.uniform(0.0, 1.0)
  base = angle * ramp_x + (1.0 - angle) * ramp_y
  canvas = Image.fromarray((base * 255).astype(np.uint8))
  draw = ImageDraw.Draw(canvas)
  for _ in range(int(rng.integers(15, 40))):
    x0, x1 = sorted(rng.integers(0, width, size=2))
    y0, y1 = sorted(rng.integers(0, height, size=2))
    shade = int(rng.integers(0, 256))
    kind = int(rng.integers(0, 3))
    if kind == 0:
      draw.rectangle([x0, y0, x1, y1], fill=shade)
    elif kind == 1:
      draw.ellipse([x0, y0, x1, y1], fill=shade)
    else:
      draw.line([x0, y0, x1, y1], fill=shade, width=int(rng.integers(1, 6)))
  array = np.asarray(canvas, dtype=np.float32) / 255.0
  array = low + array * (high - low)
  array += rng.normal(0.0, 3.0, size=array.shape)
  return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output_dir', default='data')
  parser.add_argument('--num_images', type=int, default=1000)
  parser.add_argument('--width', type=int, default=512)
  parser.add_argument('--height', type=int, default=512)
  parser.add_argument('--mixed_sizes', action='store_true',
                      help='Randomly vary the size of each image.')
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  for i in range(args.num_images):
    rng = np.random.default_rng(args.seed + i)
    width, height = args.width, args.height
    if args.mixed_sizes:
      width = int(rng.choice([256, 384, 512, 640, 768]))
      height = int(rng.choice([256, 384, 512, 640, 768]))
    path = os.path.join(args.output_dir, 'synthetic_%05d.png' % i)
    make_image(rng, width, height).save(path)
  print('Wrote %d images to %s' % (args.num_images, args.output_dir))


if __name__ == '__main__':
  main()
