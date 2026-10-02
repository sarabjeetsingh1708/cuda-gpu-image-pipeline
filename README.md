# CUDA Batch Image Pipeline

GPU-accelerated batch image processing in CUDA C++. A single execution takes a
directory of images (hundreds to thousands) and runs, entirely on the GPU:

1. **Histogram equalization** (shared-memory histogram + scan kernel + LUT)
2. **5x5 Gaussian blur** (shared-memory tiling)
3. **Sobel edge detection** (shared-memory tiling)

Images are streamed through several CUDA streams with pinned host memory so
host-to-device copies, kernels and device-to-host copies of different images
overlap. A bit-exact single-threaded CPU implementation is included for
verification and for a speedup measurement.

## Requirements

* NVIDIA GPU, CUDA Toolkit (`nvcc`) 11.6+ and a C++17 host compiler
* Python 3 with `numpy` and `Pillow` (only for the synthetic dataset script)
* Optional: `pip install cpplint` for `make lint`

Image I/O uses the header-only `stb_image` / `stb_image_write` (vendored in
`third_party/`), so there is nothing else to install.

## Build

```bash
make                # builds build/gpu_pipeline for the local GPU
make ARCH=sm_75     # or choose an architecture explicitly
```

## Run

One command does everything (build, dataset, main run, comparisons, packaging
of the proof-of-execution archive):

```bash
./run.sh                         # 1000 images of 512x512
NUM_IMAGES=2000 ./run.sh         # more data
```

Or run the program directly:

```bash
python3 scripts/generate_dataset.py --output_dir data --num_images 1000
./build/gpu_pipeline --input_dir data --output_dir out --num_images 1000 \
    --block_size 16 --num_streams 4 --cpu_baseline --log_file run.log
```

You can also point `--input_dir` at real images (for example the USC-SIPI
database); any png/jpg/bmp/pgm/tga files work, and they may have different
sizes.

### Command-line arguments

| Flag | Meaning | Default |
|------|---------|---------|
| `--input_dir DIR` | Directory with input images (required) | – |
| `--output_dir DIR` | Where result PNGs are written | `out` |
| `--num_images N` | Process at most N images (0 = all) | `0` |
| `--block_size {8,16,32}` | 2D thread-block edge length | `16` |
| `--num_streams N` | CUDA streams for copy/compute overlap | `4` |
| `--histogram_mode shared\|global` | Histogram kernel variant | `shared` |
| `--save_limit N` | Save input/equalized/edges PNGs for first N images (-1 all, 0 none) | `20` |
| `--profile_reps N` | Repetitions for the per-stage timing pass (0 = off) | `20` |
| `--cpu_baseline` | Run CPU version, verify bit-exactness, report speedup | off |
| `--log_file FILE` | Also write the log to FILE | – |

## Output

* `<name>_input.png`, `<name>_equalized.png`, `<name>_edges.png` for the first
  `--save_limit` images.
* A log containing device info, per-stage timings (CUDA events), total time for
  the batch, throughput, CPU speedup and a `PASS/FAIL` verification line.
* `run.sh` additionally writes `artifacts/` and `artifacts.tar.gz`: main log,
  `nvidia-smi` dump, sample before/after images, histogram-variant comparison,
  block-size sweep and stream-count sweep logs, plus `summary.txt`.

## Code layout

```
src/main.cu            CLI parsing, orchestration, logging, verification
src/gpu_pipeline.*     CUDA kernels + multi-stream batch executor
src/cpu_reference.*    Bit-exact CPU implementation (baseline / checker)
src/image_io.*         stb-based image loading/saving, directory listing
src/logger.*           stdout + file logging
scripts/               Synthetic dataset generator
Makefile, run.sh       Build and end-to-end run
```

The code follows the Google C++ Style Guide and passes `make lint`.

## Algorithms and design notes

* **Histogram:** `HistogramSharedKernel` gives each block a private 256-bin
  shared-memory histogram and merges it with 256 global atomics per block;
  `HistogramGlobalKernel` (selectable with `--histogram_mode global`) does one
  global atomic per pixel and suffers heavy contention on 256 counters.
* **Equalization LUT:** one 256-thread block does a Hillis-Steele inclusive
  scan of the histogram, finds the first non-zero CDF value and builds the
  lookup table with integer arithmetic only, so the GPU matches the CPU exactly.
* **Gaussian / Sobel:** each block cooperatively loads a tile plus halo into
  shared memory (clamp-to-edge borders), so every input pixel is read from
  global memory about once rather than 25 (or 9) times.
* **Overlap:** every stream owns pinned host buffers and device buffers; images
  are issued round-robin so copies and kernels of different images overlap.

## Results

Fill in from `artifacts/summary.txt` after running `./run.sh`
(GPU model, number of images, total time, images/s, CPU speedup, shared vs
global histogram time, best block size, effect of stream count).

## Lessons learned

See the project description in the submission; key points are the impact of
atomic contention, shared-memory tiling, and transfer/compute overlap.
