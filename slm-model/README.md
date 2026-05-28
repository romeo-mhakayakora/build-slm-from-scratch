# SLM Model — Pre-trained Small Language Model

A Small Language Model (~50M–70M parameters) trained from scratch on the [TinyStories](https://arxiv.org/pdf/2305.07759) dataset. Despite its size, the model generates coherent English stories by learning grammar and semantic meaning purely through next-token prediction.

---

## Mental Model — Understand This Before Touching Any Code

Before running a single cell, build this picture in your head. Everything else is just implementation detail.

### What this model actually is

This is a **mathematical function** that takes a sequence of integers and returns a probability distribution over 50,257 possible next integers. That's it. The "language understanding" is an emergent property — never explicitly programmed.

```
f(token_ids[]) → probability_distribution[50,257]
```

### How it learned

It was shown ~100 million tokens of simple children's stories. For every window of text, it was asked one question: *"what comes next?"* It got it wrong billions of times. Each time it was wrong, its 50–70 million internal numbers were nudged slightly in the right direction. After enough nudges, the numbers settled into a configuration that implicitly encodes English grammar, vocabulary, and basic logic — purely as a side effect of predicting the next word.

### The full forward pass in plain English

```
"A little girl"
      │
      ▼
① TOKENIZE
  "A" → 32   "little" → 1310   "girl" → 2576
  (words become integers via the GPT-2 BPE dictionary)
      │
      ▼
② INPUT BLOCK — two lookups, one addition
  Each token ID → 768-number vector  (what is this word?)    ← Token Embedding
  Each position → 768-number vector  (where is this word?)   ← Position Embedding
  Add them together → one 768-number vector per token        ← Input Embedding
  
  Why 768? It's the "resolution" of the representation.
  More numbers = more capacity to encode nuance.
      │
      ▼
③ PROCESSOR BLOCK — repeated 12 times
  Each token looks at every other token (that came before it)
  and decides: "how much should I borrow from each neighbor?"
  
  This is Masked Multi-Head Attention.
  "Masked" = can't look at future tokens (no cheating)
  "Multi-Head" = 12 parallel perspectives running simultaneously,
                 each learning different relationship types
                 (subject→verb, pronoun→antecedent, etc.)
  
  After attention, each token passes through a Feed-Forward layer:
  768 → 3,072 → 768  (expand to find patterns, compress back down)
  
  After 12 of these blocks, each token's 768 numbers now encode
  not just "what this word is" but "what this word means in this
  exact sentence, given everything that came before it."
      │
      ▼
④ OUTPUT BLOCK — translate back to language
  768 numbers → 50,257 numbers  (one score per vocabulary token)
  Softmax converts scores to probabilities summing to 1.0
  argmax picks the highest probability
  Look up that index in the BPE dictionary → predicted word
      │
      ▼
  "went"   (appended to input, loop repeats)
```

### The tensor shape at every stage

Think of tensor shapes as **"outer container → inner contents"**:

| Stage | Shape | Read as |
|---|---|---|
| Raw token IDs | `(B, T)` | B sequences, each T tokens long |
| After input block | `(B, T, 768)` | B sequences, T tokens, each a 768-dim vector |
| Inside attention | `(B, 12, T, T)` | B sequences, 12 heads, T×T attention scores |
| After all blocks | `(B, T, 768)` | Same shape — content enriched, shape unchanged |
| After output head | `(B, T, 50257)` | B sequences, T positions, 50,257 logit scores each |

The `T × T` attention matrix is why memory scales **quadratically** with context length — every token scores against every other token. A 1,024 token context = ~1M attention scores per head per sequence.

### Where the 50–70M parameters actually live

| Component | Formula | Approx size |
|---|---|---|
| Token Embedding Matrix | 50,257 × 768 | 38.6M |
| Position Embedding Matrix | 1,024 × 768 | 0.8M |
| Attention weights (per block) | ~2.4M | × 12 blocks = 28.8M |
| Feed-Forward weights (per block) | 8 × 768² ≈ 4.7M | × 12 blocks = 56.6M |
| Output Head | 768 × 50,257 | 38.6M |

> Notice: the vocabulary size (50,257) appears at the very start and the very end. Making it larger inflates parameters at both ends simultaneously. The middle (attention + FFN) scales with embedding dimension and number of blocks.

### Why it generates coherent text without being "taught" language

The model was never told what a noun is, what subject-verb agreement means, or that "The dog chased the **cat**, it couldn't catch **it**" requires tracking two referents across six words.

It learned all of this because **violating these rules makes the next token harder to predict correctly**. Grammar and logic are the hidden structure of language. A model that predicts the next token well has no choice but to internalize them.

This is the core insight from the TinyStories paper — and why domain-specific SLMs work: you don't need all human knowledge, you just need enough clean, consistent text from your target domain.

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

This confirms the weights loaded correctly into the model architecture with no mismatched or missing layers.

### Step 6 — Run inference

Find the inference section of the notebook. Change the `sentence` variable to any prompt you like:

```python
sentence = "A little girl went to the woods"
```

Run the cell. The model autoregressively generates a continuation — one token at a time, each predicted token appended to the input and fed back in.

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

Open the `.ipynb` file and run the cells. Skip any cell that imports from `google.colab`.

---

## Model Details

| Property | Value |
|---|---|
| Parameters | ~50M–70M |
| Architecture | GPT-2 style (Transformer decoder, trained from scratch) |
| Tokenizer | GPT-2 BPE via `tiktoken` (vocab size 50,257) |
| Training dataset | TinyStories (2M short stories, ~100M tokens) |
| Context window | 1,024 tokens |
| Transformer blocks | 12 |
| Embedding dimension | 768 |
| Attention heads | 12 |
| FFNN expansion | 768 → 3,072 → 768 (4×) |
| Optimizer | AdamW with warm-up + cosine decay |
| Hardware used | A100 GPU (Google Colab Pro) |
| Training time | ~1–2 hours |

---

## Related Resources

| Resource | Link |
|---|---|
| Full course repo | [build-slm-from-scratch](https://github.com/romeo-mhakayakora/build-slm-from-scratch) |
| TinyStories Paper | https://arxiv.org/pdf/2305.07759 |
| GPT-2 Paper | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 |
| Course Miro Board | https://miro.com/app/board/uXjVIL4LZB0=/ |
| Training Colab Notebook | https://colab.research.google.com/drive/1vZR7FUAhgGuMMzMg-JnDLPU2J1YLdSaG |
