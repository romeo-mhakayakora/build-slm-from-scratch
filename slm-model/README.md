# SLM Model — Pre-trained Small Language Model

A Small Language Model (~50M–70M parameters) trained from scratch on the [TinyStories](https://arxiv.org/pdf/2305.07759) dataset. Despite its size, the model generates coherent English stories by learning grammar and semantic meaning purely through next-token prediction.

---

## What's in This Folder

| File | Description |
|---|---|
| `Vizuara_AI_Labs_Small_Language_Model_Scratch_Final_(2) (1).ipynb` | Full notebook — data preprocessing, model architecture, training loop, and inference |
| `best_model_params.zip.part001` | Pre-trained model weights (split zip — part 1 of 2) |
| `best_model_params.zip.part002` | Pre-trained model weights (split zip — part 2 of 2) |

---

## Quickstart — Run Inference with the Pre-trained Model

### Step 1 — Clone the repo

```bash
git clone https://github.com/romeo-mhakayakora/build-slm-from-scratch.git
cd build-slm-from-scratch/slm-model
```

### Step 2 — Reassemble and unzip the model weights

The weights are split across two `.part` files. Reassemble them first:

```bash
# On Linux / Mac
cat best_model_params.zip.part001 best_model_params.zip.part002 > best_model_params.zip
unzip best_model_params.zip

# On Windows (PowerShell)
cmd /c copy /b best_model_params.zip.part001 + best_model_params.zip.part002 best_model_params.zip
Expand-Archive best_model_params.zip
```

### Step 3 — Open the notebook

The easiest way to run this is on **Google Colab**:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook** and upload the `.ipynb` file
3. Upload the reassembled `best_model_params.zip` (or the extracted weights file) to the Colab session storage
4. In the Colab runtime menu, select **Runtime → Change runtime type → T4 GPU** (A100 recommended for faster inference)

> ⚠️ **Note:** If you are running this **locally** (not on Colab), remove or skip any cell containing `from google.colab import ...` — those are Colab-specific utilities and will throw a `ModuleNotFoundError` outside of Colab. They are not needed for inference.

### Step 4 — Install dependencies

Run this cell at the top of the notebook (or in your terminal if running locally):

```bash
pip install torch tiktoken datasets
```

### Step 5 — Load the pre-trained model

Navigate to the **"Load the model"** section of the notebook and run those cells. You should see:

```
<All keys matched successfully>
```

This confirms the weights loaded correctly into the model architecture.

### Step 6 — Run inference

Find the inference section of the notebook. Change the `sentence` variable to any prompt you like:

```python
sentence = "A little girl went to the woods"
```

Run the cell. The model will autoregressively generate a continuation one token at a time.

---

## Example Outputs

**Prompt:** `"romeo mhakayakora."`
```
romeo mhakayakora. He was worried by himself. You could stop me!"
Her mom and dad looked at him. They wanted to trust him...
```

**Prompt:** `"A little girl went to the woods"`
```
A little girl went to the woods to see the birds, soft wing.
One day, Lily spotted a little girl singing in the sun...
```

---

## Running Locally (Outside Colab)

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install torch tiktoken datasets

# 3. Launch Jupyter
pip install jupyter
jupyter notebook
```

Then open the `.ipynb` file and run the cells. Skip any cell that imports from `google.colab`.

---

## Model Details

| Property | Value |
|---|---|
| Parameters | ~50M–70M |
| Architecture | GPT-2 style (Transformer decoder) |
| Tokenizer | GPT-2 BPE via `tiktoken` (vocab size 50,257) |
| Training dataset | TinyStories (2M short stories) |
| Context window | 1,024 tokens |
| Transformer blocks | 12 |
| Embedding dimension | 768 |
| Attention heads | 12 |
| Hardware used | A100 GPU (Google Colab Pro) |
| Training time | ~1–2 hours |

---

## Related Resources

| Resource | Link |
|---|---|
| Full course repo | [build-slm-from-scratch](https://github.com/romeo-mhakayakora/build-slm-from-scratch) |
| TinyStories Paper | https://arxiv.org/pdf/2305.07759 |
| GPT-2 Paper | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| Course Miro Board | https://miro.com/app/board/uXjVIL4LZB0=/ |
| Training Colab Notebook | https://colab.research.google.com/drive/1vZR7FUAhgGuMMzMg-JnDLPU2J1YLdSaG |
