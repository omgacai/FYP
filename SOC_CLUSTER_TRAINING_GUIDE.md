# SoC Cluster Training

## Access

On campus:

```bash
ssh YOUR_USER_ID@xlogin.comp.nus.edu.sg
```

Off campus, turn on the NUS VPN first:

```bash
ssh -J YOUR_USER_ID@stujump.comp.nus.edu.sg \
  YOUR_USER_ID@xlogin.comp.nus.edu.sg
```

Use the login node only for file management and Slurm commands. Run training
through Slurm, not directly on the login node. Never store passwords, SSH keys,
Hugging Face tokens, or other credentials in repositories or Slurm scripts.

## Storage and data

- Keep persistent datasets, code, caches, and experiment outputs in separate directories.
- Do not repeatedly download large datasets inside each job.
- Home-directory quota is limited. Avoid installing a complete CUDA/PyTorch
  environment there.
- Check usage with `du -sh ~/MyProject/*` before extracting large data.
- Keep train, validation, calibration, and test data separate. For multiple
  datasets, create each dataset's holdout independently to avoid leakage.
- Never use the final test split for checkpoint selection or repeated tuning.

When syncing code, preserve datasets and outputs:

```bash
rsync -avP \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".cache" \
  --exclude "Dataset" \
  --exclude "cluster_outputs" \
  --exclude "*.pt" \
  ./ YOUR_USER_ID@xlogin.comp.nus.edu.sg:~/MyProject/
```

Off campus, add:

```bash
-e "ssh -J YOUR_USER_ID@stujump.comp.nus.edu.sg"
```

Use `rsync --dry-run` before an unfamiliar command. Avoid `--delete` unless
remote deletion is intentional.

## Runtime environment

A working approach is an Apptainer PyTorch image such as:

```text
docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
```

Use the container's PyTorch, torchvision, CUDA, and cuDNN. Install only small
missing packages into node-local job storage:

```bash
JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER-myproject-$SLURM_JOB_ID}"
PYTHON_PACKAGES="$JOB_TMP/python-packages"
mkdir -p "$PYTHON_PACKAGES"

apptainer exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" \
  python3 -m pip install --quiet --no-cache-dir \
    --target "$PYTHON_PACKAGES" PACKAGE_NAME

export APPTAINERENV_PYTHONPATH="$PYTHON_PACKAGES:$PROJECT_DIR"
```

Use persistent model caches:

```bash
export APPTAINER_CACHEDIR="$PROJECT_DIR/.apptainer-cache"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
```

Do not reinstall PyTorch into the temporary package layer unless the container
itself is unsuitable.

## Slurm details to verify

The working resource pattern for our run was:

```bash
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nv:1
```

Cluster configuration can change. Verify the node, partition, GPU resource name,
and time limit before reusing those lines:

```bash
sinfo -N -n xgpd0 -o '%N %P %t %G'
sinfo -o '%P %a %l %D %G'
```

The `gpu` partition may have a three-hour limit. A longer requested time can
make submission fail even if the node appears in that partition.

Inside the job, bind the project and temporary directory:

```bash
apptainer exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --bind "$JOB_TMP:$JOB_TMP" \
  "$CONTAINER" \
  python3 -u "$PROJECT_DIR/train.py" ...
```

Use `set -euo pipefail`, include `%j` in log filenames, and validate before
submission:

```bash
bash -n slurm/train_gpu.sbatch
sbatch slurm/train_gpu.sbatch
```

## GPU and batch size

- Print `nvidia-smi` at the beginning of the job.
- Print the PyTorch version, CUDA availability, GPU name, and selected device
  from the training program.
- `sacct`'s `MaxRSS` is CPU resident memory, not peak GPU VRAM.
- Run a short memory test before increasing physical batch size.
- Use gradient accumulation to raise effective batch size without proportionally
  increasing peak activation memory.

Example:

```text
physical batch 16 × accumulation 4 = effective batch 64
```

For a frozen CLIP ViT-L/14 on the TITAN V, physical batch 16 worked in this
project. This is not guaranteed for another architecture or image size.

## Monitoring

```bash
squeue -u "$USER"
squeue -j JOB_ID -o '%.18i %.9T %.10M %.19S %R'
tail -f slurm/logs/JOB_NAME-JOB_ID.out
tail -n 100 slurm/logs/JOB_NAME-JOB_ID.err
```

Pressing `Ctrl+C` during `tail -f` stops only the log viewer; it does not cancel
the Slurm job.

After completion:

```bash
sacct -j JOB_ID \
  --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Take note:

- `PENDING (Resources)` normally means the requested GPU is busy.
- `COMPLETED` with exit code `0:0` means the job succeeded.
- `OUT_OF_MEMORY` normally requires a smaller physical batch.
- `TIMEOUT` requires resuming from a checkpoint or using an eligible longer
  partition.
- A quiet log during a long validation pass does not necessarily mean the job
  is stuck.
- Do not move a working setup to an untested GPU node merely to avoid a queue;
  some node/container combinations may fail.

## Checkpoints and evaluation

Use a new output directory for each experiment and retain:

- The best validation checkpoint.
- A resumable checkpoint containing model, optimizer, scheduler, epoch, and
  best score.
- Machine-readable metrics and training arguments.
- Slurm output and error logs.

Save resumable state after every epoch. Never overwrite the last known working
model while testing a new configuration.

For multiple dataset domains, report validation metrics separately and select
checkpoints with a source-aware score. Aggregate accuracy can hide poor
performance on a smaller dataset.

Download the complete experiment directory after training:

```bash
rsync -avP \
  YOUR_USER_ID@xlogin.comp.nus.edu.sg:~/MyProject/cluster_outputs/EXPERIMENT/ \
  cluster_results/EXPERIMENT/
```

Off campus, add the same ProxyJump option used for code synchronization.

The local evaluation environment must recreate the same architecture and
preprocessing recorded in the checkpoint. A frozen backbone is still required
for inference; “frozen” only means its weights were not updated during training.

## Recommended sequence

1. Run data-split and checkpoint-loading tests.
2. Run a one-epoch smoke test.
3. Confirm GPU use, memory stability, loss movement, validation output, and
   checkpoint creation.
4. Start the full job in a new output directory.
5. Monitor until several training intervals are stable; the SSH session can be
   closed afterward.
6. Compare source-specific validation and robustness results.
7. Calibrate on a separate calibration split.
8. Evaluate the untouched test split only after the model choice is fixed.

If the same failure repeats without new information, stop resubmitting and
diagnose the first reproducible error instead.
