# Run CubiCasa5K + CubiGraph5K locally on macOS

This folder now contains `cubicasa5k_cubigraph_macos.ipynb`. It is a separate local copy; the original Colab notebook remains unchanged.

## Setup

```bash
cd ~/Downloads/FYP
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_macos.txt
jupyter lab
```

Create `input/` and add a plan image as `input/floorplan.png`, or edit `INPUT_IMAGE` in the notebook.

## macOS execution

The notebook selects Apple Silicon MPS when PyTorch exposes it and otherwise uses CPU. If a legacy CubiCasa operation is unsupported on MPS, the inference cell retries on CPU automatically. It does not require CUDA.

## Important limitation

CubiGraph5K deterministically derives relations from semantically annotated SVG room/door geometry. The notebook's SVG is generated from CubiCasa predictions as an experimental bridge; visually inspect its polygons and relation graph before treating it as ground truth.
