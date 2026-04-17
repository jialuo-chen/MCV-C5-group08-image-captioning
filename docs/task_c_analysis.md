# Task C — Captioning Model Weakness Analysis

## Model Under Analysis

**ViT-base (frozen) + Qwen3.5-2B (LoRA)** — Best Optuna trial #27  
Test metrics: BLEU-1=0.679, BLEU-2=0.431, ROUGE-L=0.443, METEOR=0.420  
Evaluated on 7,542 VizWiz test images with per-sample metrics (sorted worst-to-best).

## Identified Problems

### 1. Vocabulary Collapse and Repetitive Captions (MAJOR)

The model has an extremely limited output vocabulary of **2,789 unique words** vs **16,672 in ground-truth references** — an 83% vocabulary gap. It falls into a small set of template phrases:

| Repeated Prediction | Occurrences |
|---------------------|-------------|
| "A computer screen with a blue background and white text." | 104x |
| "A computer screen with a message on it." | 53x |
| "A computer screen with a black background and white text." | 34x |
| "A black screen with nothing on it." | 26x |
| "A can of food is on a counter top." | 24x |

31 unique captions each appear more than 5 times. 76% of all captions are 6–11 words — they all follow the same "A [adjective] [object] [preposition] [location]" template. The type-token ratio (0.035) is significantly lower than that of references (0.042), confirming the model generates monotonous language.

**Root cause**: The small LoRA adapter (r=4) combined with a frozen encoder constrains the model's expressiveness. The training objective (cross-entropy per token) rewards "safe" high-probability tokens over diverse, specific vocabulary.

### 2. Hallucination of Specific Names and Titles (MAJOR)

**292 predictions** contain quoted text (titles, names, brands). Of these, **268 (92%)** do not match any reference — the model fabricates specific content.

| Type | Count | Example Prediction → Actual |
|------|-------|----------------------------|
| DVD/movie titles | 36 | "The Man Who Shot Liberty Valance" → Bob Willis CD |
| TV show names | 20 | "The Big Bang Theory" → Hannah Montana DVD |
| Book/magazine titles | 29 | "The Complete Guide to Cook's Illustrated" → Kids science craft |
| Software/website | 55 | "security website" → Satellite map browser |
| Game titles | 7 | "The Sims 3" → Baseball card |
| Other | 116 | Various fabricated names |

The brand "Campbell's" appears in 61 predictions but is wrong in 17 cases (28%), being applied to Dr Pepper cans, Ocean Spray juice, and other unrelated items.

**Root cause**: The model memorizes strong co-occurrences from training data (can + red label → Campbell's, rectangular object → DVD → popular title) and generates them as default fills without visual grounding.

### 3. Generic Object Descriptions (MAJOR)

The model overwhelmingly defaults to hypernym-level descriptions:

- **560 predictions** contain the generic word "food" — only **1** uses a specific brand name
- **148 predictions** say "a can of food" instead of identifying the actual product
- Common pattern: "A box of frozen food" instead of "Barilla whole grain penne pasta"
- "A white device" instead of "power strip", "A black electronic device" instead of specific gadgets

Top-4 caption starts show extreme templating:
- "A computer screen with" — 453 times
- "A person is holding" — 376 times
- "A close up of" — 214 times
- "A can of food" — 113 times

**Root cause**: High-frequency category words dominate the training distribution. The model learns that generic descriptions are "safe" and rarely penalized by the loss function compared to risking a wrong specific term.

### 4. Object Misidentification

The model misidentifies objects even when it attempts specificity:

| Image | Prediction | Ground Truth |
|-------|-----------|--------------|
| val_00001532 | "George Washington" portrait | **Abraham Lincoln** (five dollar bill) |
| val_00003209 | "pink and white **shoe**" | Pink/brown **Coach handbag** |
| val_00004109 | "person wearing green shirt" | **Bread** with brown parts |
| val_00003354 | "black electronic device with power cord" | **Barbecue grill** by Holland |
| val_00003308 | "alcohol hand sanitizer" | Package of **Crayola sidewalk chalk** |
| val_00005962 | "bag of chips" | Package of **10mg wrapped pills** |
| val_00001683 | "person sitting in a car" | **Wooden surface** with paper and comb |

**Root cause**: The 224×224 resolution of ViT-base loses critical fine-grained details. The frozen encoder also limits the model's ability to adapt visual features to the captioning task.

### 5. Difficulty with VizWiz-Specific Image Quality

VizWiz images are captured by visually impaired users. 107 test images have references explicitly noting quality issues ("too severe to recognize"). Beyond these:

- Many images have **extreme blur, camera flash glare, finger-over-lens**, and **dark/overexposed** scenes
- The model produces 26 copies of "A black screen with nothing on it" and 8 copies of "A bright light shining on a dark background"
- On degraded images, METEOR averages 0.324 (vs 0.420 overall)
- Single-reference images (3.6% of test set) have dramatically lower METEOR (0.222 vs 0.427 for multi-reference)

**Root cause**: Limited training signal from degraded images. The model hasn't learned to describe what partial information is visible in poor photos.

### 6. Inability to Read Text in Images

Labels, screens, and documents are described structurally but text is never read:
- "A white piece of paper with black text" instead of reading the actual content
- "A computer screen with a message" appearing 53 times for wildly different screens
- Screen content is never identified beyond vague descriptions

**Root cause**: Architectural limitation — frozen ViT-base at 224×224 was not trained for OCR. This cannot be fully fixed with synthetic data but could be improved with text-aware training examples.

## Research Question

> **Can targeted synthetic data — generated with Stable Diffusion for underrepresented and commonly confused categories — improve captioning accuracy on VizWiz?**

Based on this analysis, the three most impactful problems addressable by synthetic data are:

1. **Vocabulary collapse / genericity**: Synthetic images with specific, detailed captions can teach the model to produce diverse vocabulary instead of defaulting to "food", "device", "screen"
2. **Hallucinated titles/brands**: Targeted examples of DVDs, books, cans with correct-but-varied descriptions can break the memorized co-occurrence patterns
3. **Object misidentification**: Paired examples of visually similar but semantically different objects (Lincoln vs Washington, bags vs shoes) can teach fine-grained distinctions

We hypothesize synthetic training data will:
1. Increase METEOR and BLEU scores on the VizWiz test set
2. Reduce the rate of hallucinated quoted text (from 92% wrong to lower)
3. Improve vocabulary diversity (increase type-token ratio)

Both YES and NO outcomes are informative: if synthetic data doesn't help, the domain gap between clean SD3.5 outputs and degraded VizWiz photos is too large, or the errors stem from architectural limitations (LoRA rank, frozen encoder, resolution) rather than data scarcity.

## Target Categories for Synthetic Data

| Category | Weakness Addressed | Scale of Problem |
|----------|-------------------|-----------------|
| Generic food/products | 560 generic "food" predictions | High — largest error source |
| DVD/book/media | 92 hallucinated titles | High — worst METEOR impact |
| Brand-name products | Only 1 specific brand prediction | High — near-zero specificity |
| Electronics/devices | "white device", "black electronic device" | Medium |
| Currency/bills | President misidentification | Low — specific but memorable |
| Accessories | Bag/shoe confusion | Low |
| Degraded scenes | 107 quality-issue images | Medium — hard for SD to replicate |
| Computer screens | 453 identical "screen with" captions | High — massive repetition |
