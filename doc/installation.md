## Installation
### A. Clone repository

```bash
# clone repository
git clone https://github.com/DekuLiuTesla/CityGaussian.git
cd CityGaussian
```

### B. Create virtual environment

#### Option 1: Using uv (Recommended)

```bash
# install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# create virtual environment
uv venv --python 3.9
source .venv/bin/activate
```

#### Option 2: Using conda

```bash
# create virtual environment
conda create -yn gspl python=3.9 pip
conda activate gspl
```

### C. Install PyTorch
* Tested on `PyTorch==2.0.1` and `PyTorch==2.5.1`
* You must install the one matching your CUDA version (check with `nvcc --version`)

#### For CUDA 11.8

```bash
# Using uv
uv pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Using conda/pip
pip install -r requirements/pyt201_cu118.txt
```

#### For CUDA 12.4+ (RTX 40-series and newer)

```bash
# Using uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Using conda/pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### D. Install requirements

```bash
# Using uv
uv pip install -r requirements.txt

# Using conda/pip
pip install -r requirements.txt
```

For pose and 3DGS joint optimization, please directly jump to this [document](../doc/vggt_x.md).

### E. Install additional package for CityGaussian

```bash
# Using uv
uv pip install -r requirements/CityGS.txt

# Using conda/pip
pip install -r requirements/CityGS.txt
```

Note that here we use modified version of Trim2DGS rasterizer, so as to resolve [impulse noise problem](https://github.com/hbb1/2d-gaussian-splatting/issues/174) under street views. This version also avoids interference from out-of-view surfels.

### F. Build CUDA submodules

For CUDA 12.x, you need to build submodules with `--no-build-isolation` to allow torch access during compilation:

```bash
# Using uv (recommended for CUDA 12.x)
uv pip install -e ./submodules/diff-gaussian-rasterization --no-build-isolation
uv pip install -e ./submodules/diff-surfel-rasterization-city --no-build-isolation
uv pip install -e ./submodules/diff-trim-gaussian-rasterization --no-build-isolation
uv pip install -e ./submodules/simple-knn --no-build-isolation

# Using conda/pip
pip install -e ./submodules/diff-gaussian-rasterization
pip install -e ./submodules/diff-surfel-rasterization-city
pip install -e ./submodules/diff-trim-gaussian-rasterization
pip install -e ./submodules/simple-knn
```

### G. Fix setuptools (if using PyTorch Lightning)

Newer setuptools versions (v70+) removed `pkg_resources` which PyTorch Lightning requires:

```bash
# Using uv
uv pip install "setuptools>=61.0,<70"

# Using conda/pip
pip install "setuptools>=61.0,<70"
```
