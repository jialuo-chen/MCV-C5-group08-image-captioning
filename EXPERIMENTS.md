# Experiment Plan — Image Captioning on VizWiz

> **Chosen alternatives**: Encoder = **ResNet-50**, Decoder = **LSTM**
> **Baseline**: ResNet-18 + GRU + char-level, no attention

---

## 1 — Experiment Matrix

We need to isolate the effect of **each individual modification** to the baseline.
Every experiment changes **exactly one variable** from its reference, except for the final "best combo" runs.

### Phase A — Single-Variable Ablations (no sweep needed)

These experiments use **fixed hyperparameters** (the baseline defaults) so each comparison is fair.
The only thing that changes between paired experiments is the variable under test.

| ID | Name | Encoder | Decoder | Tokenizer | Attention | Config file | Run? |
|----|------|---------|---------|-----------|-----------|-------------|------|
| **A0** | **Baseline** | ResNet-18 | GRU | char | ✗ | `baseline.yaml` | ✅ |
| **A1** | Encoder swap | **ResNet-50** | GRU | char | ✗ | `resnet50_gru.yaml` | ✅ |
| **A2** | Decoder swap | ResNet-18 | **LSTM** | char | ✗ | `baseline_lstm.yaml` | ✅ |
| **A3** | Text → subword | ResNet-18 | GRU | **subword** | ✗ | `baseline_subword.yaml` | ✅ |
| **A4** | Text → word | ResNet-18 | GRU | **word** | ✗ | `baseline_word.yaml` | ✅ |
| **A5** | + Attention | ResNet-18 | GRU | char | **Bahdanau** | `baseline_attention.yaml` | ✅ |

**Comparisons obtained:**

```
A0 vs A1  →  Impact of encoder   (ResNet-18 vs ResNet-50);  A1 [ResNet-50]
A0 vs A2  →  Impact of decoder   (GRU vs LSTM);  A0 [GRU]
A0 vs A3  →  Impact of tokenizer (char vs subword); A3 [subword]
A0 vs A4  →  Impact of tokenizer (char vs word); A4 [word]
A0 vs A5  →  Impact of attention  (none vs Bahdanau); A5 [Bahdanau]
A3 vs A4  →  Subword vs word (text representation comparison); A3 [subword]
```

### Phase B — Combined Modifications (no sweep needed)

Once we know each component's individual effect, combine the chosen alternatives:

| ID | Name | Encoder | Decoder | Tokenizer | Attention | Config file | Run? |
|----|------|---------|---------|-----------|-----------|-------------|------|
| **B1** | Enc+Dec | ResNet-50 | LSTM | char | ✗ | `resnet50_lstm.yaml` | ✅ |
| **B2** | Enc+Dec+Word | ResNet-50 | LSTM | word | ✗ | `resnet50_lstm_word.yaml` | ✅ |
| **B3** | Enc+Dec+Subword | ResNet-50 | LSTM | subword | ✗ | `resnet50_lstm_subword.yaml` | ✅ |
| **B4** | Enc+Dec+Attn | ResNet-50 | LSTM | char | Bahdanau | `resnet50_lstm_attention.yaml` | ✅ |
| **B5** | Full combo (word) | ResNet-50 | LSTM | word | Bahdanau | `resnet50_lstm_word_attn.yaml` | ✅ |
| **B6** | Full combo (subword) | ResNet-50 | LSTM | subword | Bahdanau | `resnet50_lstm_subword_attn.yaml` | ✅ |

**Comparisons obtained:**

```
B1 vs A1  →  Impact of decoder when encoder is already ResNet-50; B1 [LSTM win]
B1 vs A2  →  Impact of encoder when decoder is already LSTM; B1 [ResNet-50 win]
B1 vs B4  →  Impact of attention on top of ResNet-50 + LSTM; B4 [with attention win]
B4 vs B5  →  Impact of word tokenizer on the full pipeline; B5 [word win]
B4 vs B6  →  Impact of subword tokenizer on the full pipeline; B6 [subword win]
B5 vs B6  →  Word vs subword in the full pipeline; B6 [subword win]
A0 vs B6  →  Full delta: baseline → best combo; B6 [ΔBLEU-1: ↑0.1583; ΔBLEU-2: ↑0.1540; ΔROUGE-L: ↑0.8020; ΔMETEOR: ↑0.8820]
```

### Phase GRU — Combined Modifications (no sweep needed)

More experiments testing the ResNet-50 + GRU combo, to confirm the trends hold across decoders:

| ID | Name | Encoder | Decoder | Tokenizer | Attention | Config file | Run? |
|----|------|---------|---------|-----------|-----------|-------------|------|
| **GRU1** | Enc+Dec+Word | ResNet-50 | GRU | word | ✗ | `resnet50_gru_word.yaml` | ✅ |
| **GRU2** | Enc+Dec+Subword | ResNet-50 | GRU | subword | ✗ | `resnet50_gru_subword.yaml` | ✅ |
| **GRU3** | Enc+Dec+Attn | ResNet-50 | GRU | char | Bahdanau | `resnet50_gru_attention.yaml` | ✅ |
| **GRU4** | Full combo (word) | ResNet-50 | GRU | word | Bahdanau | `resnet50_gru_word_attn.yaml` | ✅ |
| **GRU5** | Full combo (subword) | ResNet-50 | GRU | subword | Bahdanau | `resnet50_gru_subword_attn.yaml` | ✅ |

### Phase C — Hyperparameter Sweep (WandB Bayesian search)

Sweep is needed to **fairly optimize** the final chosen architectures and for learning-rate-sensitive configs.

| ID | Name | What to sweep | Base config | Why |
|----|------|--------------|-------------|-----|
| **C1** | Sweep: Baseline | lr, batch_size, hidden_size, embed_size, dropout, num_layers, optimizer | `baseline.yaml` | Ensure baseline is at its best before comparing |
| **C2** | Sweep: Best Phase-B config | lr, batch_size, hidden_size, embed_size, dropout, num_layers, optimizer, weight_decay, scheduler | Best from Phase B | Find optimal HPs for the best architecture |

**Sweep parameters (shared template):**

```yaml
parameters:
  training.lr:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.01
  training.batch_size:
    values: [32, 64, 128]
  training.optimizer:
    values: [adam, adamw]
  training.weight_decay:
    values: [0.0, 0.0001, 0.001]
  training.scheduler:
    values: [null, cosine]
  decoder.hidden_size:
    values: [256, 512, 1024]
  decoder.embed_size:
    values: [256, 512]
  decoder.num_layers:
    values: [1, 2]
  decoder.dropout:
    values: [0.0, 0.1, 0.3, 0.5]
```

---

## 2 — What Needs / Doesn't Need a Sweep

### ✅ No sweep needed (Phase A + B)

All single-variable ablations and combinations **must** use identical hyperparameters so the comparison is valid. Default training config:

```
lr: 0.001 | optimizer: adam | batch_size: 64 | hidden_size: 512
embed_size: 512 | num_layers: 1 | dropout: 0.0 | grad_clip: 5.0
epochs: 20 | early_stopping_patience: 5 | scheduler: null
```

> **Exception**: word-level experiments use `epochs: 50`, `early_stopping_patience: 10`, `optimizer: adamw` because word-level tokenizers converge slower with a much larger vocabulary. This is a practical necessity, not a tuning choice.

### 🔍 Sweep needed (Phase C)

- **C1**: to establish the best the baseline can achieve (otherwise the combined model might look better just because of HP differences).
- **C2**: to find the best HP setting for the final architecture  and report the strongest possible numbers.

---

## 3 — Comparison Matrix

This table shows which experiment pairs answer which question:

| Question | Compare | Variable isolated |
|----------|---------|-------------------|
| Effect of **encoder** (ResNet-18→50) | A0 vs A1 | Encoder only |
| Effect of **decoder** (GRU→LSTM) | A0 vs A2 | Decoder only |
| Effect of **text** (char→subword) | A0 vs A3 | Tokenizer only |
| Effect of **text** (char→word) | A0 vs A4 | Tokenizer only |
| **Subword vs word** head-to-head | A3 vs A4 | Tokenizer only |
| Effect of **attention** | A0 vs A5 | Attention only |
| Encoder + decoder **combined** | A0 vs B1 | Encoder + decoder |
| Attention on **strong backbone** | B1 vs B4 | Attention on ResNet-50+LSTM |
| Best text repr on **full pipeline** | B5 vs B6 | Word vs subword (everything else equal) |
| **Full improvement over baseline** | A0 vs B5/B6 | Everything |
| **Optimized baseline vs optimized best** | C1-best vs C2-best | Arch quality after fair tuning |

---

## 4 — Config Files Needed

### Already exist (6 experiments ready to run):

| Config | Used by |
|--------|---------|
| `baseline.yaml` | A0 |
| `resnet50_gru.yaml` | A1 |
| `baseline_lstm.yaml` | A2 |
| `baseline_subword.yaml` | A3 |
| `baseline_word.yaml` | A4 |
| `baseline_attention.yaml` | A5 |
| `resnet50_lstm.yaml` | B1 |
| `resnet50_lstm_attention.yaml` | B4 |

### Need to create (4 new configs):

| Config | Derives from | Changes |
|--------|-------------|---------|
| `resnet50_lstm_word.yaml` | `resnet50_lstm.yaml` | tokenizer → word, max_length → 100, epochs → 50, patience → 10, optimizer → adamw |
| `resnet50_lstm_subword.yaml` | `resnet50_lstm.yaml` | tokenizer → subword, vocab_size → 4000, max_length → 50 |
| `resnet50_lstm_word_attn.yaml` | `resnet50_lstm_word.yaml` | attention.enabled → true, attention.type → bahdanau |
| `resnet50_lstm_subword_attn.yaml` | `resnet50_lstm_subword.yaml` | attention.enabled → true, attention.type → bahdanau |

### Sweep configs (2 new):

| Config | Base |
|--------|------|
| `sweep_baseline.yaml` | `baseline.yaml` |
| `sweep_best.yaml` | whichever Phase B config wins |

---

## 5 — Suggested Execution Order

```
PHASE A — Single-variable ablations (can be run in parallel if multi-GPU)
─────────────────────────────────────────────────────────────────────────
  1.  A0  baseline.yaml                 ← anchor for all comparisons
  2.  A1  resnet50_gru.yaml             ← encoder effect
  3.  A2  baseline_lstm.yaml            ← decoder effect
  4.  A3  baseline_subword.yaml         ← subword effect
  5.  A4  baseline_word.yaml            ← word effect
  6.  A5  baseline_attention.yaml       ← attention effect

       ⬇ Analyze Phase A results → pick best tokenizer & confirm attention helps; best tokenizer: subword, attention: yes

PHASE B — Combined architectures
─────────────────────────────────────────────────────────────────────────
  7.  B1  resnet50_lstm.yaml            ← encoder + decoder
  8.  B2  resnet50_lstm_word.yaml       ← + word
  9.  B3  resnet50_lstm_subword.yaml    ← + subword
 10.  B4  resnet50_lstm_attention.yaml  ← + attention
 11.  B5  resnet50_lstm_word_attn.yaml  ← full combo word
 12.  B6  resnet50_lstm_subword_attn.yaml ← full combo sub

       ⬇ Analyze Phase B results → pick final best architecture; R50 + LSTM + sub + Bah [bests]

PHASE C — Hyperparameter optimization (Bayesian sweep, ~20 runs each)
─────────────────────────────────────────────────────────────────────────
 13.  C1  sweep_baseline.yaml           ← optimize baseline
 14.  C2  sweep_best.yaml               ← optimize best Phase B config

       ⬇ Final comparison: C1-best vs C2-best
```

---

## 6 — Reporting Template

For each experiment, collect:

| Metric | Source |
|--------|--------|
| BLEU-1, BLEU-2, ROUGE-L, METEOR | `evaluate` command |
| Encoder params / Decoder params / Total | `experiment_log.json` → model_info |
| FLOPs | `experiment_log.json` → model_info |
| Training time | `experiment_log.json` → summary |
| Best epoch | `experiment_log.json` → summary |
| Loss curves (train + val) | `experiment_log.json` → epochs array or WandB |
| Qualitative samples | `visualize` command (5-10 images per model) |

### Final results table (to fill in):

| Exp | Encoder | Decoder | Tok | Attn | BLEU-1 | BLEU-2 | ROUGE-L | METEOR | Params | FLOPs | Infer. Time |
|-----|---------|---------|-----|------|--------|--------|---------|--------|--------|-------|-------------|
| A0 | R18 | GRU | char | ✗ | 0.4505 | 0.2242 | 0.3314 | 0.2836 | 13M | 3.63G | 33s |
| A1 | R50 | GRU | char | ✗ | 0.4543 | 0.2526 | 0.3342 | 0.3086 | 26M | 8.18G | 32s |
| A2 | R18 | LSTM | char | ✗ | 0.3882 | 0.1759 | 0.2721 | 0.2817 | 13M | 3.63G | 32s |
| A3 | R18 | GRU | sub | ✗ | 0.5588 | 0.3078 | 0.3656 | 0.3132 | 17M | 3.71G | 32s |
| A4 | R18 | GRU | word | ✗ | 0.5046 | 0.2853 | 0.3640 | 0.3129 | 22M | 3.81G | 32s |
| A5 | R18 | GRU | char | Bah | 0.4589 | 0.2657 | 0.3519 | 0.3195 | 14M | 7.51G | 32s |
| B1 | R50 | LSTM | char | ✗ | 0.5289 | 0.3064 | 0.3600 | 0.3283 | 27.8M | 8.18G | 390s |
| B2 | R50 | LSTM | word | ✗ | 0.5585 | 0.3465 | 0.4070 | 0.3554 | 38.0M | 8.37G | 250s |
| B3 | R50 | LSTM | sub | ✗ | 0.6205 | 0.3795 | 0.4104 | 0.3622 | 31.8M | 8.26G | 240s |
| B4 | R50 | LSTM | char | Bah | 0.5000 | 0.3025 | 0.3740 | 0.3454 | 32.6M | 17.34G| 625s |
| B5 | R50 | LSTM | word | Bah | 0.5609 | 0.3496 | 0.4131 | 0.3696 | 42.8M | 17.53G| 320s |
| B6 | R50 | LSTM | sub | Bah | 0.6088 | 0.3782 | 0.4116 | 0.3718 | 36.7M | 17.41G| 312s |
| C1★ | R18 | GRU | char | ✗ | 0.5020 | 0.2885 | 0.3572 | 0.3254 | 23.3M | 3.63G | 39s |
| C2★ | _best_: R50 | _best_: LSTM | _best_: sub | _best_: Bah | | | | | | | |

> ★ = after HP sweep
