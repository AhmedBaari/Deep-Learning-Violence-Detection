These instructions apply to **all GitHub Copilot activity** in this repository, especially in **`.ipynb` notebooks** for the HMDB51 violence detection project.[1][2][3][4]

***

## 1. High-level goals

- Implement and maintain a **journal-grade deep learning pipeline** for **violence detection** on the **HMDB51 Fight** dataset (8 classes, ~75,900 images) using **PyTorch only**.
- Use a **Transformer-based backbone (ViT/Swin/DeiT, default: ViT-Base)**, plus **Neural Structured Learning**, **advanced augmentations**, and **adversarial training** to reach **87–90% test accuracy** (baseline VGG-16 is 71%).
- All heavy work happens inside **Jupyter (`.ipynb`) notebooks**, executed in **VS Code** connected to a **DGX with 8× NVIDIA H200 GPUs**.
- Notebooks must be **restart-safe** and **reproducible**, with all expensive results cached under `novelty_files/`.

When in doubt, **prioritize correctness, clarity, and restart safety over cleverness or brevity**.[5][6][1]

***

## 2. Important project context for Copilot

### 2.1 Core constraints

- **Framework**: PyTorch only.  
  - Never use TensorFlow, Keras, or other DL frameworks.
- **Hardware**: 8× NVIDIA H200 GPUs, CUDA 12.8, NCCL backend.
- **Parallelism**: Use **`torch.distributed` + `DistributedDataParallel` (DDP)** for all serious training.
- **Environment**: VS Code + Jupyter notebooks (remote SSH to DGX).[7][8]
- **Dataset**: HMDB51 Fight, images organized as:
  - `./data/hmdb51/<class_name>/*.jpg`
- **Filesystem contract (critical)**:
  - All outputs must be under:
    ```text
    novelty_files/
      configs/
      splits/
      features/
      graphs/
      checkpoints/
      logs/
      metrics/
      visualizations/
    ```
  - **Never write to the project root** outside `novelty_files/`.

### 2.2 Attached reference materials Copilot should use

Whenever Copilot needs to understand design, architecture, or working code, it should **look at these files first** instead of inventing new patterns:

- **Base paper & implementation (reference / behavior ground truth)**  
  - `base-paper.pdf`  
  - `1_base_paper_implementation.pdf`  
    - Use this as a **reference for working PyTorch code** and dataset handling.
- **Novelty plan / research design**  
  - `Novelty-Implementation-Deep-Research.pdf`
- **Notebook definitions (the “spec” for each notebook)**  
  - `01_data_and_splits.md`  
  - `02_vit_baseline.md`  
  - `03_nsl_and_graphs.md`  
  - `04_training_pipeline.md`  
  - `05_advanced_augmentations.md`  
  - `06_adversarial_and_regularization.md`  
  - `07_ablation_studies.md`  
  - `08_evaluation_and_visualization.md`
- **Global guides**  
  - `EXECUTIVE_SUMMARY_start_here.md`  
  - `README_master_index.md`  
  - `QUICK_START_guide.md`  
  - `FILE_INDEX_complete_map.md`  
  - `DELIVERABLES_summary.md`

**Instruction**:  
Before generating new code or changing notebook structure, **consult the relevant `.md` notebook spec and the base implementation PDF**, and **match the described behavior and interfaces** as closely as possible.[2][9][4]

***

## 3. How Copilot should behave in `.ipynb` notebooks

These rules are specific to Jupyter / VS Code notebook workflows.[8][10][7]

### 3.1 Cell-by-cell workflow

For any non-trivial notebook:

1. **Work one cell at a time**:
   - Propose code for a single cell.
   - The user will run that cell.
   - If an error occurs, help **fix that exact cell** before suggesting new cells.

2. **After each successful cell**, Copilot should:
   - Encourage printing:
     - Tensor shapes.
     - First few rows of data.
     - Sample images or outputs where relevant.
   - Encourage simple assertions (e.g., `assert x.shape[0] > 0`).

3. **Avoid generating entire notebooks in one go.**  
   Favor incremental development over large monolithic suggestions.[10][7]

### 3.2 Restart-safe & caching behavior

Every expensive step (data splits, feature extraction, graph building, training, ablations, evaluation) must:

- **Check for existing artifacts** in `novelty_files/`.
- **Load from disk and skip recomputation** if the artifact exists and is compatible.
- Only recompute if:
  - The file is missing, or
  - The user explicitly asks to recompute.

Preferred pattern:

```python
from pathlib import Path

output_path = Path("novelty_files/features/train_features.pt")
output_path.parent.mkdir(parents=True, exist_ok=True)

if output_path.exists():
    print(f"✓ Found cached features at {output_path}, loading instead of recomputing.")
    features = torch.load(output_path)
else:
    print("No cached features found. Computing features now (this may take a while)...")
    # ... compute features ...
    torch.save(features, output_path)
    print(f"✓ Saved features to {output_path}")
```

**Copilot must default to this pattern** for any step that uses a lot of compute or GPU time.[11][12]

### 3.3 DDP and 8-GPU usage

When inside training notebooks (especially corresponding to `04_training_pipeline.md`, `06_adversarial_and_regularization.md`, `07_ablation_studies.md`):

- Use **DistributedDataParallel** with:
  - `torch.distributed.init_process_group(backend="nccl")`
  - `torch.cuda.set_device(rank)`
  - `DistributedSampler` for datasets.

- Include clear, commented setup code that:
  - Reads `RANK`, `WORLD_SIZE` env vars or uses `torch.distributed.get_rank()`.
  - Asserts that `WORLD_SIZE == 8` for main training runs.

- Use **only one rank (usually rank 0)** for:
  - Logging.
  - Metric aggregation and printing.
  - Saving checkpoints and metrics to disk.

- Copilot should **not** switch to DataParallel or single-GPU unless user explicitly says so.[13][14][15]

### 3.4 Notebook style & explanations

Copilot should:

- Generate **verbose, well-commented code**, not one-liners.
- Prefer **notebook-specific utility functions** defined in the same notebook or a local helper file, rather than large, abstract frameworks.
- Add **docstrings** and inline comments explaining:
  - What this cell is doing.
  - Why it is needed for the research pipeline (e.g., “this step constructs the NSL graph used to regularize the model toward smooth predictions between neighbors”).
- When suggesting plots, **also save them** to `novelty_files/visualizations/` with descriptive filenames.

Example expectations:

```python
def train_one_epoch(...):
    """
    Train the model for a single epoch.

    This function:
    - Iterates over all training batches.
    - Computes the main classification loss and additional NSL loss.
    - Backpropagates and updates model parameters.
    - Returns the average loss for logging.
    """
    # Clear structure, clear comments
```

***

## 4. How Copilot should use existing code & guides

### 4.1 Reusing base implementation

- When implementing:
  - Dataset loading.
  - Transform pipelines.
  - Metric computation.
  - Basic training loops.

Copilot should **look into**:

- `1_base_paper_implementation.pdf`  
- Any existing `.py` or `.ipynb` files derived from that implementation.

Then:

- **Prefer adapting those patterns** over generating completely new ones.
- Preserve:
  - Proven-working pre-processing pipelines.
  - Label encodings.
  - Evaluation quirks that match the base paper.
- Only deviate when required for:
  - ViT/NSL/adversarial specific changes.
  - Distributed training.

### 4.2 Following provided notebook specs

Each `.md` notebook file describes **exactly what that notebook should do**. Copilot must:

- Treat each `0X_*.md` as the **specification** for its corresponding `.ipynb`.
- Follow:
  - The **cell structure** (roughly).
  - The **sequence of operations**.
  - The **saved artifact names and locations**.
- Keep filenames, directory paths, and artifact names **consistent** with those specs to ensure all notebooks interoperate correctly.

If there is ambiguity:

- Prefer what is written in:
  1. `README_master_index.md`
  2. The specific `0X_*.md` spec
  3. Base implementation PDF

***

## 5. Error handling & auto-fixing behavior

The user wants Copilot to **actively help solve errors in notebooks**, not just generate code once.

When errors occur:

1. **Explain the error** in simple terms:
   - What went wrong.
   - Which line or concept is the root cause.

2. **Suggest a minimal fix** that:
   - Preserves existing structure.
   - Respects the filesystem contract and DDP pattern.
   - Doesn’t silently change global behavior unless obviously necessary.

3. **Update the cell** rather than rewriting the entire notebook:
   - Provide a corrected version of the problematic cell.
   - Include a short explanation in comments about the fix.

4. For recurring patterns (e.g., shape mismatches, CUDA memory, DDP sync issues), propose **safeguards**:
   - Shape checks and informative `assert` messages.
   - Optional smaller batch sizes and gradient accumulation.
   - `dist.barrier()` placements for synchronizing processes.

***

## 6. Project-specific coding standards

- **No TensorFlow/Keras**.
- **No global state hidden in magic**:
  - If something is important across cells, save it to disk (`novelty_files/*`) explicitly.
- **No writing outside `novelty_files/`** (except reading from `./data/hmdb51/`).
- **Path handling**:
  - Use `pathlib.Path` and `mkdir(parents=True, exist_ok=True)` for directories.
- **Logging**:
  - Write important logs as text or JSON under `novelty_files/logs/` and `novelty_files/metrics/`.
- **Reproducibility**:
  - Encourage setting seeds (`torch`, `numpy`, `random`) at the start of each notebook.
  - Avoid non-deterministic flags unless explicitly disabled.

***

## 7. What to optimize for

When Copilot has multiple valid options, it should prefer:

1. **Accuracy** (reach 87–90% test accuracy).
2. **Stability and debuggability** (clear errors, restart-safe).
3. **Readability and learning value** (helps a student understand).

Over:

- Clever but opaque code.
- Micro-optimizations that complicate logic.
- Experimental/untested techniques.

***

## 8. Example behaviors Copilot should emulate

- In a new notebook cell:
  - Summarize the step in a short Markdown heading.
  - Write **clear, commented PyTorch code** that:
    - Reads from or writes to `novelty_files/`.
    - Prints sample outputs and shapes.
    - Includes assertions and checks.

- When user asks for “implement NSL loss here”:
  - Look for NSL description in `03_nsl_and_graphs.md` and `Novelty-Implementation-Deep-Research.pdf`.
  - Implement `virtual_adversarial_loss`, `l2_neighbor_loss`, and the combined NSL term in a way that matches the described formulas and usage.

- When user asks “continue training from here”:
  - Look for existing checkpoints in `novelty_files/checkpoints/`.
  - Load best/latest checkpoint if present.
  - Resume training with consistent optimizer and scheduler state.

***

## 9. Scope and limitations

- These instructions apply primarily to:
  - `.ipynb` notebooks.
  - Supporting `.py` files used by those notebooks.
- Copilot should **not**:
  - Introduce new frameworks or major dependencies without user confirmation.
  - Change the overall directory structure.
  - Ignore the attached `.md` specs and PDFs when they conflict with its own guesses.

If Copilot is unsure, it should:

- Prefer asking the user a **clarifying question** in a notebook markdown cell or comment, rather than making a large, potentially breaking change.

***