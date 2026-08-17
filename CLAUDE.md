# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CityGaussian is a large-scale scene reconstruction system using 3D/2D Gaussian Splatting. Built on [Gaussian Lightning](https://github.com/yzslab/gaussian-splatting-lightning) with PyTorch Lightning.

## Build & Installation

**Important**: This project requires CUDA-compiled submodules. For CUDA 12.x (GTX 40-series and newer), the submodules need `<cstdint>` includes in header files.

**Always use `uv` for package management. Never use `pip` or `pip install` directly - always use `uv add` or `uv pip install`.**

```bash
# Create environment
uv venv --python 3.9
source .venv/bin/activate

# Install PyTorch matching your CUDA version
# For CUDA 11.8:
uv add torch==2.0.1 torchvision torchaudio --index https://download.pytorch.org/whl/cu118
# For CUDA 12.4:
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu124

# Install base requirements
uv add -r requirements.txt

# Install CityGaussian specific packages
uv add -r requirements/CityGS.txt

# Build CUDA submodules (requires --no-build-isolation for torch access during build)
uv pip install -e ./submodules/diff-gaussian-rasterization --no-build-isolation
uv pip install -e ./submodules/diff-surfel-rasterization-city --no-build-isolation
uv pip install -e ./submodules/diff-trim-gaussian-rasterization --no-build-isolation
uv pip install -e ./submodules/simple-knn --no-build-isolation

# Fix setuptools version (newer versions removed pkg_resources needed by lightning)
uv pip install "setuptools<70"
```

## Common Commands

### Training Pipeline
```bash
# 1. Train coarse model
python main.py fit --config configs/citygsv2_lfls_coarse_sh2.yaml -n experiment_name

# 2. Partition scene and assign data
python utils/partition_citygs.py --config_path configs/citygsv2_lfls_sh2_trim.yaml --force

# 3. Parallel fine-tuning
python utils/train_citygs_partitions.py -n experiment_name

# 4. Merge checkpoints
python utils/merge_citygs_ckpts.py outputs/experiment_name
```

### Evaluation & Rendering
```bash
# Test/evaluate model
python main.py test --config outputs/experiment_name/config.yaml --save_val --test_speed

# Mesh extraction
python utils/gs2d_mesh_extraction.py outputs/experiment_name --voxel_size 0.05

# Render video
python render.py model.pth --camera-path-filename camera.json --output-path output.mp4

# Web viewer
gs-viewer model.pth --port 8080
```

### Testing
```bash
python -m unittest discover tests -v
python -m unittest tests.vanilla_gaussian_model_test -v
```

### Quick Installation Test
```bash
# Download test dataset (Tanks & Temples truck scene, ~180MB)
wget "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip" -O data/tandt.zip
unzip data/tandt.zip -d data/
ln -s images data/tandt/truck/images_2

# Train for 3000 steps (~2 min on RTX 4080)
python main.py fit --data.path data/tandt/truck \
    --data.parser.class_path Colmap \
    --data.parser.init_args.down_sample_factor 2 \
    --data.parser.init_args.down_sample_rounding_mode ceil \
    --trainer.max_steps 3000 -n test_truck

# View result
gs-viewer outputs/test_truck/checkpoints/*.ckpt --port 8080
```

## Architecture

### Entry Points
- `main.py` → `internal/entrypoints/gspl.py` - Main CLI (fit/val/test/predict)
- `render.py` - Video rendering with camera paths
- `viewer.py` - Web-based model viewer

### Core Modules (internal/)
- `gaussian_splatting.py` - Main LightningModule for training
- `models/` - Gaussian model variants (vanilla, 2D, mip, deformable, appearance)
- `renderers/` - Pluggable rendering backends (vanilla, gsplat, partition_lod)
- `dataset.py` - DataModule with async caching for large datasets
- `dataparsers/` - Dataset loaders (Colmap, Blender, NSVF, Nerfies)
- `density_controllers/` - Gaussian densification strategies
- `callbacks.py` - Training callbacks (SaveGaussian, etc.)

### Submodules
- `diff-gaussian-rasterization` → `diff_gaussian_rasterization` - Standard 3DGS rasterizer
- `diff-surfel-rasterization-city` → `diff_trim_surfel_rasterization` - Modified surfel rasterizer for street views
- `diff-trim-gaussian-rasterization` → `diff_trim_gaussian_rasterization` - Trimmed Gaussian variant
- `simple-knn` → `simple_knn` - KNN implementation

## Configuration

YAML configs in `/configs/` define model, data, renderer, and optimizer settings. Override via CLI:
```bash
python main.py fit --config configs/base.yaml --trainer.max_steps 30000 --data.parser.image_scale 2
```

Output structure: `outputs/<experiment_name>/config.yaml`, `checkpoints/`, `cameras.json`

## Key Patterns

- **Lightning-based**: Training uses LightningModule/DataModule pattern
- **Modular renderers**: Swap renderers via config (vanilla, gsplat, feature_3dgs)
- **Scene partitioning**: Large scenes split into blocks for parallel training
- **Caching**: Async image caching to manage memory on large datasets

## Allowed Commands

The following commands are pre-approved and can be run without confirmation:

- `python -c "import ..."` - Python import tests
- `python main.py --help` - CLI help
- `python main.py fit/validate/test/predict ...` - Training and evaluation
- `python -m unittest ...` - Running tests
- `uv add ...` - Installing packages
- `uv pip install ... --no-build-isolation` - Building CUDA extensions
- `gs-viewer ...` - Web viewer
- `wget ...` - Download files
- `curl ...` - Download files
- `gdown ...` - Download from Google Drive
- `unzip ...` - Extract archives
