#!/usr/bin/env bash
# Builds the project, generates a dataset (if needed), runs the pipeline on the
# whole dataset in a single execution and collects proof-of-execution
# artifacts into artifacts/ and artifacts.tar.gz.
#
# Environment overrides: NUM_IMAGES (default 1000), IMAGE_SIZE (default 512),
# ARCH (passed to make), DATA_DIR (default data).
set -euo pipefail
cd "$(dirname "$0")"

NUM_IMAGES="${NUM_IMAGES:-1000}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
DATA_DIR="${DATA_DIR:-data}"
ARTIFACTS=artifacts

make -j"$(nproc)" ${ARCH:+ARCH=$ARCH}

if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
  python3 scripts/generate_dataset.py --output_dir "$DATA_DIR" \
    --num_images "$NUM_IMAGES" --width "$IMAGE_SIZE" --height "$IMAGE_SIZE"
fi

rm -rf "$ARTIFACTS" out
mkdir -p "$ARTIFACTS/samples" out
(nvidia-smi || echo "nvidia-smi not available") > "$ARTIFACTS/nvidia_smi.txt" 2>&1
ls "$DATA_DIR" | wc -l > "$ARTIFACTS/dataset_file_count.txt"

# 1) Main run: whole dataset, CPU baseline + verification, sample images saved.
./build/gpu_pipeline --input_dir "$DATA_DIR" --output_dir "$ARTIFACTS/samples" \
  --num_images "$NUM_IMAGES" --block_size 16 --num_streams 4 \
  --histogram_mode shared --save_limit 10 --cpu_baseline \
  --log_file "$ARTIFACTS/run_main.log"

# 2) Naive global-atomics histogram, for comparison with the shared version.
./build/gpu_pipeline --input_dir "$DATA_DIR" --output_dir out \
  --num_images "$NUM_IMAGES" --histogram_mode global --save_limit 0 \
  --log_file "$ARTIFACTS/run_global_histogram.log"

# 3) Block-size sweep.
for bs in 8 16 32; do
  ./build/gpu_pipeline --input_dir "$DATA_DIR" --output_dir out \
    --num_images "$NUM_IMAGES" --block_size "$bs" --save_limit 0 \
    --log_file "$ARTIFACTS/sweep_block_${bs}.log"
done

# 4) Stream-count sweep (1 stream = no copy/compute overlap).
for ns in 1 2 4 8; do
  ./build/gpu_pipeline --input_dir "$DATA_DIR" --output_dir out \
    --num_images "$NUM_IMAGES" --num_streams "$ns" --save_limit 0 \
    --log_file "$ARTIFACTS/sweep_streams_${ns}.log"
done

grep -h "^\[gpu\]\|^\[speedup\]\|^\[verify\]" "$ARTIFACTS"/*.log \
  > "$ARTIFACTS/summary.txt" || true
tar -czf artifacts.tar.gz "$ARTIFACTS"
echo "Done. Upload artifacts.tar.gz; commit the artifacts/ folder."
