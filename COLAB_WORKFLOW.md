# Google Colab Execution Workflow

This guide details how to work on the **LLMs & VLMs from Scratch** project using Google Colab.

---

## 1. Setting Up Google Colab

### A. Selecting Hardware Accelerator
1. In Colab, go to **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (free tier) or **A100 / L4 GPU** (Colab Pro).
3. Ensure Python 3 runtime is selected.

### B. Mounting Google Drive (For Saving Checkpoints & Datasets)
Colab instances are ephemeral. If your session disconnects, any downloaded dataset or model checkpoint stored locally will be erased. Always save checkpoints to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
CHECKPOINT_DIR = '/content/drive/MyDrive/scratch_series_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
```

---

## 2. Working with the GitHub Repository in Colab

### Workflow Option A: Direct Notebook Open via GitHub (Recommended for Reading & Prototyping)
You can open any notebook directly in Colab by replacing `github.com` with `colab.research.google.com/github/`:
- URL format:
  `https://colab.research.google.com/github/Uchihakamal1816/llms-and-vlms-from-scratch/blob/main/llm/notebooks/<notebook_name>.ipynb`

### Workflow Option B: Git Clone with Personal Access Token (Recommended for Development)
To push your code changes from Colab back to GitHub:

1. Generate a GitHub Personal Access Token (PAT) with `repo` permissions at:
   https://github.com/settings/tokens
2. In Colab, clone your repo using your credentials securely:

```python
import os
from google.colab import userdata

# Tip: Store your token in Colab Secrets (the key icon on the left sidebar) as 'GITHUB_TOKEN'
try:
    token = userdata.get('GITHUB_TOKEN')
except Exception:
    token = input("Enter your GitHub Token: ")

username = "Uchihakamal1816"
repo_name = "llms-and-vlms-from-scratch"

!git clone https://{username}:{token}@github.com/{username}/{repo_name}.git
%cd {repo_name}
!git config user.name "Uchihakamal1816"
!git config user.email "saikamalnannuri@gmail.com"
```

3. After completing your implementation in Colab:
```python
!git status
!git add .
!git commit -m "Implement phase step in notebook"
!git push origin main
```

---

## 3. Recommended GPU Memory Management in Colab
When training transformer models from scratch on Colab's T4 GPU (16 GB VRAM):

| Parameter | Recommended Setting for T4 GPU |
| :--- | :--- |
| **Precision** | `torch.bfloat16` or `torch.float16` with `torch.cuda.amp.autocast()` |
| **Batch Size** | Start small (`batch_size = 4` to `16`) |
| **Sequence Length** | `256` to `512` during early prototyping; `1024` for pretraining |
| **Gradient Accumulation** | Use `gradient_accumulation_steps = 4` or `8` to simulate larger batch sizes |
| **Cache Clearing** | Call `torch.cuda.empty_cache()` between experiments |

---

## 4. Suggested Datasets for Scratch Training in Colab

### For LLM Pretraining:
1. **TinyShakespeare** (~1 MB text): Ideal for sub-10 minute sanity tests.
2. **TinyStories** (~500 MB text): High quality synthetic English stories with simple vocabulary. A 10M–30M parameter model achieves coherent, grammatical outputs within 1–2 hours on a T4.
3. **OpenWebText (Subsampled)**: For broader web-scale language modeling.

### For VLM Pretraining & Instruction Tuning:
1. **Flickr8k / Flickr30k**: Lightweight image-caption dataset ideal for Stage 1 projector alignment.
2. **COCO Captions (Subsampled)**: Image description pairs.
3. **LLaVA-Instruct-150K (Tiny subset)**: For multimodal instruction tuning.
