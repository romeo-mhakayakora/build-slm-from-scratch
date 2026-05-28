## SLM Model Inference Guide

This folder contains a trained checkpoint and the notebook used to run inference:

- `best_model_params.pt`
- `best_model_params.zip.part001`
- `best_model_params.zip.part002`
- `best_model_params.zip`
- `Vizuara_AI_Labs_Small_Language_Model_Scratch_Final_(2) (1).ipynb`

## 1) Requirements

- Python 3.10+ (3.11 also works)
- `torch`
- `tiktoken`

Install:

```bash
pip install torch tiktoken
```

## 2) If You Need To Rebuild The Zip

Only needed if you downloaded split parts and do not have `best_model_params.pt` yet.

```powershell
Get-Content .\best_model_params.zip.part* -Encoding Byte -ReadCount 0 | Set-Content .\best_model_params.zip -Encoding Byte
Expand-Archive .\best_model_params.zip -DestinationPath .\best_model_params -Force
```

## 3) Important: Architecture Must Match Training

Your checkpoint was trained with:

- `vocab_size=50257`
- `block_size=128`
- `n_layer=6`
- `n_head=6`
- `n_embd=384`
- `dropout=0.1`

If these do not match in code, loading will fail or produce poor results.

## 4) Minimal Inference Steps

Use the model class definitions from the notebook (`GPTConfig`, `GPT`) and run:

```python
import torch
import tiktoken

enc = tiktoken.get_encoding("gpt2")

config = GPTConfig(
    vocab_size=50257,
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
)

model = GPT(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load("best_model_params.pt", map_location=torch.device(device)))
model = model.to(device)
model.eval()

sentence = "A little girl went to the woods"
context = torch.tensor(enc.encode_ordinary(sentence), dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    y = model.generate(context, max_new_tokens=200)

print(enc.decode(y.squeeze().tolist()))
```

When loading works, you should see:

```text
<All keys matched successfully>
```

## 5) Prompt Examples

- `romeo mhakayakora.`
- `A little girl went to the woods`

## 6) Output Quality Expectations

Based on your notebook outputs, the model can continue story-style text but may drift on long generations:

- repetition
- abrupt topic switching
- occasional grammar instability

This is normal for a small model trained from scratch.

## 7) Local Jupyter vs Colab

This line is Colab-only and will fail locally:

```python
from google.colab import runtime
```

Local error:

```text
ModuleNotFoundError: No module named 'google.colab'
```
