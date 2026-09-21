"""Read-only EGTR compatibility probe: no installs and no CUDA compilation."""
import importlib.metadata
import platform
from pathlib import Path
import sys

print('Architecture:',platform.machine(),flush=True)
print('Python:',sys.executable,flush=True)
if platform.machine()!='x86_64': raise SystemExit('Use an x86-64 GPU node, e.g. xgph1')
import torch
print('Torch:',torch.__version__,'CUDA available:',torch.cuda.is_available(),flush=True)
for name in ('transformers','timm','ninja','scipy','torchvision'):
    try: print(name,importlib.metadata.version(name),flush=True)
    except importlib.metadata.PackageNotFoundError: print(name,'MISSING',flush=True)
# Importing upstream normally compiles CUDA on import. Disable only its CUDA
# availability test during this API compatibility probe; do not alter torch.
import transformers
from transformers import file_utils
original=file_utils.is_torch_cuda_available
file_utils.is_torch_cuda_available=lambda:False
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'third_party/egtr'))
try:
    from model.egtr import DetrForSceneGraphGeneration
    print('EGTR Python imports: PASS. CUDA extension and checkpoint forward still untested.')
finally:
    file_utils.is_torch_cuda_available=original
