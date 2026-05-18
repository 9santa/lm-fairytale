# Tiny Fairytale Language Model in C++/LibTorch

A small educational language-modeling project that implements and trains autoregressive models for TinyStories-style fairy-tale generation.

The goal of this project is **not** to fine-tune an existing LLM. Instead, it builds the core language-modeling pipeline from the ground up in C++:

- data loading and batching
- character/byte-level tokenization
- SentencePiece BPE tokenization
- bigram language model baseline
- decoder-only Transformer language model
- causal self-attention
- multi-head attention
- feedforward MLP blocks
- residual connections and LayerNorm
- autoregressive sampling with temperature and top-k

LibTorch is used for tensors, autograd, modules, and optimizers. The model architecture and training pipeline are implemented directly in C++.

---

## Motivation

This project was built as a learning-focused implementation of a small language model. The main question was:

> How much does a model improve as we move from local token statistics to contextual self-attention, and then from character-level tokens to learned subword tokens?

The project progresses through three increasingly capable models:

1. **Bigram LM** — learns only immediate next-token transition statistics.
2. **Character-level Transformer** — learns contextual next-character prediction using causal self-attention.
3. **SentencePiece Transformer** — uses learned BPE subword tokens, improving effective context length and generation quality.

---

## Dataset

The experiments use the TinyStories V2 GPT-4 text files:

```text
data/raw/TinyStoriesV2-GPT4-train.txt
data/raw/TinyStoriesV2-GPT4-valid.txt
```

For the initial experiments, a subset was used:

```text
Train chars: 5,000,000
Valid chars:   500,000
```

The dataset already provides a train/validation split, so no custom split is needed.

---

## Autoregressive LM Formally

Given a token sequence

$$
x_1, x_2, \dots, x_T
$$

a language model factorizes the sequence probability as

$$
P(x_1, x_2, \dots, x_T)
=
\prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})
$$

The training task is next-token prediction.

For a token window:

```text
x = [t0, t1, t2, ..., t127]
y = [t1, t2, t3, ..., t128]
```

the model receives `x` and predicts `y`.

For a batch:

```text
x:      [batch_size, context_len]
y:      [batch_size, context_len]
logits: [batch_size, context_len, vocab_size]
```

The logits and targets are flattened before applying cross-entropy:

```text
logits:  [B, T, V] -> [B*T, V]
targets: [B, T]    -> [B*T]
where: B = batch_size, T = context_len, V = vocab_size
```

The loss is:

$$
\mathcal{L}
=
-\frac{1}{BT}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log P_\theta(y_{b,t} \mid x_{b,\leq t})
$$

---

## Models

### 1. Bigram Language Model

The bigram model makes the approximation:

$$
P(x_t \mid x_1, \dots, x_{t-1})
\approx
P(x_t \mid x_{t-1})
$$

It only uses the immediately previous token.

Implementation-wise, the model is just a learned lookup table:

```text
Embedding(vocab_size, vocab_size)
```

For each input token, the embedding row is interpreted as logits over the next token.

This model is intentionally weak, but it is useful as a sanity check and a baseline:

- verifies tokenization
- verifies batching
- verifies loss computation
- verifies backprop and optimizer
- provides a baseline for comparison

---

### 2. Character / Byte-Level Transformer

The Transformer replaces the bigram lookup table with a contextual function:

$$
h_t = f_\theta(x_1, \dots, x_t)
$$

$$
\text{logits}_t = W_{\text{out}} h_t
$$

The implemented decoder-only Transformer is:

```text
idx [B, T]
  -> token + position embeddings [B, T, d_model]
  -> TransformerBlock 1 [B, T, d_model]
  -> TransformerBlock 2 [B, T, d_model]
  -> final LayerNorm [B, T, d_model]
  -> Linear head [B, T, vocab_size]
```

Each Transformer block:

```text
x = x + MultiHeadCausalSelfAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

The attention module is causal, so position `t` can only attend to positions `<= t`.

---

### 3. SentencePiece BPE Transformer

The character-level model spends a lot of capacity learning spelling and word boundaries. To improve this, we replace char-level tokenization with SentencePiece BPE subword-tokenization.

With character-level tokenization:

```text
once upon a time
-> o n c e _ u p o n _ a _ t i m e
```

With SentencePiece BPE:

```text
once upon a time
-> ▁once ▁upon ▁a ▁time
```

The architecture stays the same. Only the tokenizer and vocabulary size change.

SentencePiece statistics from the 5M-character training subset:

```text
SP vocab size:       1024
Train token ids:     1,442,752
Valid token ids:     144,227
Train chars/token:   3.4656
Valid chars/token:   3.46676
```

So the sequence becomes about **3.47x shorter** than character-level tokenization. With the same context length of 128 tokens, the model sees roughly:

```text
128 * 3.47 ≈ 444 characters
```

instead of only 128 characters.

---

## Architecture Details

### Token and Position Embeddings

Input token IDs have shape:

```text
idx: [B, T]
```

Token embeddings produce:

```text
tok_emb: [B, T, d_model]
```

Learned position embeddings produce:

```text
pos_emb: [T, d_model]
```

The final input representation is:

```text
x = tok_emb + pos_emb
```

Broadcasting applies the same positional vectors across the batch:

```text
[B, T, d_model] + [T, d_model] -> [B, T, d_model]
```

---

### Causal Self-Attention

For input:

```text
x: [B, T, d_model]
```

one attention head computes:

```text
Q = x * W_q
K = x * W_k
V = x * W_v
```

with shapes:

```text
Q, K, V: [B, T, head_dim]
```

Attention scores:

$$
S = \frac{QK^\top}{\sqrt{d_{\text{head}}}}
$$

Shape:

```text
scores: [B, T, T]
```

A lower-triangular causal mask prevents tokens from attending to future positions by making them equal -∞.

Then:

$$
A = \text{softmax}(S_{\text{masked}})
$$

$$
Y = A V
$$

Shape:

```text
Y: [B, T, head_dim]
```

Multi-head attention runs several heads in parallel and concatenates their outputs:

```text
n_heads * [B, T, head_dim] -> [B, T, d_model]
```

A final linear layer mixes information across heads.

---

### Feedforward MLP

Each Transformer block contains a position-wise MLP:

```text
d_model -> 4 * d_model -> d_model
```

For example:

```text
128 -> 512 -> 128
256 -> 1024 -> 256
```

The MLP does not mix sequence positions. Attention handles communication between tokens; the MLP performs local nonlinear processing at each token position.

---

## Training Setup

Common settings used in the main experiments:

```text
Optimizer:      AdamW
Gradient clip:  1.0
Context length: 128
Batch size:     32
Dropout:        0.1
Dataset:        5M train chars / 500k valid chars
```

Generation uses autoregressive sampling with:

```text
temperature
top-k filtering
```

For SentencePiece experiments, typical sampling settings were:

```text
temperature = 0.7
top_k = 30
```

---

## Results

### Summary Table

| Model | Tokenizer | Config | Steps | Valid loss | Notes |
|---|---|---:|---:|---:|---|
| Bigram LM | char/byte | vocab=59 | 3,000 | 2.4769 | Learns local transition statistics only |
| Transformer | char/byte | 2L, d=128, h=4, ctx=128 | 3,000 | 1.3541 | Much better than bigram; still fake words |
| Transformer | char/byte | 2L, d=128, h=4, ctx=128 | 10,000 | 1.0858 | Better phrase structure; spelling still weak |
| Transformer | SentencePiece BPE | 2L, d=128, h=4, ctx=128, vocab=1024 | 5,000 | 2.9007 | Cleaner words; token loss not directly comparable |
| Transformer | SentencePiece BPE | 2L, d=128, h=4, ctx=128, vocab=1024 | 10,000 | 2.6464 | Better coherence; rough char-normalized loss ≈ 0.763 |
| Transformer | SentencePiece BPE | 2L, d=256, h=4, ctx=128, vocab=1024 | 5,000 | 2.3063 | Best run so far; rough char-normalized loss ≈ 0.665 |

### Important note about comparing losses

Losses across tokenizers are **not directly comparable** because one character token and one SentencePiece token represent different amounts of text.

For SentencePiece:

```text
valid chars/token ≈ 3.46676
```

A rough character-normalized loss estimate is:

$$
L_{\text{char}} \approx \frac{L_{\text{SPM token}}}{3.46676}
$$

For the `d_model=256` SentencePiece run:

```text
SPM token loss:       2.3063
rough loss per char:  2.3063 / 3.46676 ≈ 0.665
```

This is substantially better than the character-level Transformer loss of about `1.0858` per character token.

---

## Training Curves

### Bigram LM

```text
[step 0]    train loss = 4.68456, valid loss = 4.68656
[step 200]  train loss = 4.38437, valid loss = 4.38549
[step 400]  train loss = 4.11346, valid loss = 4.10946
[step 600]  train loss = 3.86518, valid loss = 3.86553
[step 800]  train loss = 3.64679, valid loss = 3.64494
[step 1000] train loss = 3.44917, valid loss = 3.44879
[step 1200] train loss = 3.28024, valid loss = 3.27501
[step 1400] train loss = 3.13076, valid loss = 3.12728
[step 1600] train loss = 2.99915, valid loss = 2.99553
[step 1800] train loss = 2.88494, valid loss = 2.87990
[step 2000] train loss = 2.78726, valid loss = 2.78673
[step 2200] train loss = 2.70532, valid loss = 2.70173
[step 2400] train loss = 2.63253, valid loss = 2.62917
[step 2600] train loss = 2.56928, valid loss = 2.56714
[step 2800] train loss = 2.51767, valid loss = 2.51829
[step 2999] train loss = 2.47585, valid loss = 2.47685
```

### Character Transformer, d_model=128, 3k steps

```text
[step 0]    train loss = 4.26170, valid loss = 4.26197
[step 200]  train loss = 2.33800, valid loss = 2.33022
[step 400]  train loss = 2.25016, valid loss = 2.24702
[step 600]  train loss = 2.16086, valid loss = 2.15615
[step 800]  train loss = 2.04470, valid loss = 2.03025
[step 1000] train loss = 1.89214, valid loss = 1.88649
[step 1200] train loss = 1.77270, valid loss = 1.76056
[step 1400] train loss = 1.69650, valid loss = 1.67166
[step 1600] train loss = 1.61834, valid loss = 1.59667
[step 1800] train loss = 1.56081, valid loss = 1.55039
[step 2000] train loss = 1.50914, valid loss = 1.50602
[step 2200] train loss = 1.47255, valid loss = 1.45351
[step 2400] train loss = 1.42752, valid loss = 1.43019
[step 2600] train loss = 1.41459, valid loss = 1.40535
[step 2800] train loss = 1.38272, valid loss = 1.37057
[step 2999] train loss = 1.34599, valid loss = 1.35413
```

### SentencePiece Transformer, d_model=128, 10k steps

```text
[step 0]    train loss = 7.11230, valid loss = 7.11102
[step 500]  train loss = 4.10943, valid loss = 4.09247
[step 1000] train loss = 3.68822, valid loss = 3.67899
[step 1500] train loss = 3.45421, valid loss = 3.48888
[step 2000] train loss = 3.31953, valid loss = 3.32702
[step 2500] train loss = 3.21594, valid loss = 3.21703
[step 3000] train loss = 3.12128, valid loss = 3.16529
[step 3500] train loss = 3.05011, valid loss = 3.07483
[step 4000] train loss = 2.99710, valid loss = 3.01719
[step 4500] train loss = 2.91654, valid loss = 2.95715
[step 5000] train loss = 2.86170, valid loss = 2.92870
[step 5500] train loss = 2.82503, valid loss = 2.85259
[step 6000] train loss = 2.75784, valid loss = 2.85474
[step 6500] train loss = 2.73867, valid loss = 2.81171
[step 7000] train loss = 2.71061, valid loss = 2.76940
[step 7500] train loss = 2.66824, valid loss = 2.74805
[step 8000] train loss = 2.62820, valid loss = 2.73106
[step 8500] train loss = 2.61675, valid loss = 2.70969
[step 9000] train loss = 2.58772, valid loss = 2.69317
[step 9500] train loss = 2.56083, valid loss = 2.66287
[step 9999] train loss = 2.53098, valid loss = 2.64644
```

### SentencePiece Transformer, d_model=256, 5k steps

```text
[step 0]    train loss = 7.10809, valid loss = 7.10705
[step 500]  train loss = 3.60677, valid loss = 3.61661
[step 1000] train loss = 3.25491, valid loss = 3.29025
[step 1500] train loss = 3.00986, valid loss = 3.05849
[step 2000] train loss = 2.79425, valid loss = 2.83729
[step 2500] train loss = 2.59925, valid loss = 2.67578
[step 3000] train loss = 2.45788, valid loss = 2.56557
[step 3500] train loss = 2.35150, valid loss = 2.49945
[step 4000] train loss = 2.28253, valid loss = 2.42730
[step 4500] train loss = 2.19886, valid loss = 2.34826
[step 4999] train loss = 2.12932, valid loss = 2.30632
```

---

## Sample Outputs

### Bigram LM

The bigram model learns local character statistics, but not coherent language:

```text
once upon a timery. o t54dor542�vely. cowad
ted t5qund. bum g tr. fr thaledr th��s<m! o oy o g he. d.
```

### Character Transformer

The character Transformer produces recognizable story-like fragments, but still invents many word-shaped errors:

```text
once upon a time, there was a big tree and said, "i will be named and the bird not with me, in a small dog named broke and dad."
the little boy was sad and the cat was scary and the face and the sky. the fun was too saw the tree and wanted to her toys.
```

### SentencePiece Transformer, d_model=128

The SentencePiece model produces much cleaner word-level text, but still has repetition and weak semantics:

```text
once upon a time, there was a big, impressful little boy named tim. tim and his mom went back to the garden. the next day, tim found a big, shiny toy car. tim was so happy and shared his ball. tim was very happy.
```

### SentencePiece Transformer, d_model=256

The larger SentencePiece model produces the best sample so far:

```text
once upon a time, there was a little girl named lily. she was very patient and always wanted to play in the park and play with her friends. one day, she found a big, soft bed on the ground. she picked it up and saw something shiny. the girl was curious. it was very excited to see what was inside the house. she asked her mom if she could go up to the store. but, she knew it was time for her mom and dad. the girl became a good thing. they all wanted to play inside it, but they were not miserable.
```

The remaining issues are no longer spelling-level failures. They are mostly semantic and story-planning mistakes.

---

## What Worked

### 1. Bigram baseline was useful

The bigram model was weak, but it verified that the pipeline was correct:

- data loading
- tokenization
- random batching
- cross-entropy loss
- optimizer
- generation

### 2. Self-attention significantly improved validation loss

The character Transformer beat the bigram baseline very quickly:

```text
Bigram, 3000 steps:             valid loss ≈ 2.48
Char Transformer, 200 steps:    valid loss ≈ 2.33
Char Transformer, 3000 steps:   valid loss ≈ 1.35
```

This shows that contextual self-attention is much stronger than first-order transition statistics.

### 3. SentencePiece improved effective context length

The SentencePiece tokenizer reduced the sequence length by about `3.47x`, making the same 128-token context cover much more raw text.

### 4. Increasing d_model helped

Moving from `d_model=128` to `d_model=256` gave a large improvement:

```text
SPM d=128, 10k steps: valid loss ≈ 2.65
SPM d=256,  5k steps: valid loss ≈ 2.31
```

This suggests the smaller SentencePiece Transformer was capacity-limited.

---

## Limitations

- The project uses only a 5M-character subset of TinyStories for most experiments.
- The model is still very small compared to modern LLMs.
- The model has no instruction tuning or RLHF.
- The model has weak long-range story planning.
- Raw token losses cannot be directly compared across different tokenizers.
- The current generation loop does not yet stop automatically at `<|endoftext|>`.
- Evaluation is mostly validation loss plus qualitative samples; no automated story-quality metric is implemented yet.

---

## Possible Next Steps

- Add stop-on-`<|endoftext|>` generation.
- Train on more of the TinyStories dataset.
- Try `context_len = 256`.
- Compare SentencePiece vocab sizes: `512`, `1024`, `2048`.
- Compare `n_heads = 4` vs `n_heads = 8` for `d_model=256`.
- Try deeper models: `n_layers = 4`.
- Add checkpoint resume support.
- Add CSV logging for losses.
- Plot training/validation loss curves.
- Add loss-per-character / bits-per-character reporting.
- Add top-p sampling.
- Add a small CLI for training and sampling.
- Add unit tests for tokenizer, causal masking, and tensor shapes.

---

## Build and Run

This project uses CMake, LibTorch, and SentencePiece.

Example build:

```bash
cmake -B build -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build build
```

Example binaries used during development:

```bash
./build/train_bigram
./build/train_transformer
./build/test_sentencepiece
./build/train_transformer_spm
```

For Neovim/clangd, generating `compile_commands.json` is useful:

```bash
ln -sf build/compile_commands.json .
```

---

## Project Status

Implemented:

- [x] data loading
- [x] train/validation loading from TinyStories files
- [x] character/byte-level tokenizer
- [x] random sequence batching
- [x] bigram LM baseline
- [x] token and position embeddings
- [x] single-head causal self-attention
- [x] multi-head causal self-attention
- [x] feedforward MLP
- [x] Transformer block
- [x] TransformerLM
- [x] Transformer training loop
- [x] autoregressive generation
- [x] SentencePiece BPE tokenizer
- [x] SentencePiece Transformer training

Not implemented yet:

- [ ] stop generation at `<|endoftext|>`
- [ ] checkpoint resume
- [ ] automated plots
- [ ] full-dataset training
- [ ] CLI config files
- [ ] unit tests

---

## Main Takeaway

The project demonstrates a clear progression:

```text
bigram statistics
    -> contextual self-attention
    -> learned subword tokenization
    -> larger hidden representations
```

Each step produced a measurable and qualitative improvement.

The strongest result so far is the SentencePiece BPE Transformer with `d_model=256`, which produces coherent TinyStories-like text and substantially improves rough character-normalized loss over the Bigram model and character-level Transformer.
