# Task C — Captioning Model Weakness Analysis

## Model Under Analysis

**ViT-base (frozen) + Qwen3.5-2B (LoRA)** — Best Optuna trial #27  
Test metrics: BLEU-1=0.679, BLEU-2=0.431, ROUGE-L=0.443, METEOR=0.420

## Identified Problems

### 1. Object Misidentification

The model confuses visually similar objects with different identities.

| Image | Prediction | Ground Truth |
|-------|-----------|--------------|
| val_00001532 | "a person holding a picture of **George Washington**" | Five dollar bill with **Abraham Lincoln** |
| val_00003209 | "a pink and white **shoe** with a heart on it" | A pink, brown, and white **Coach handbag** |
| val_00007011 | "a DVD case for **The Big Bang Theory**" | A DVD case with a man in a yellow suit (unrelated show) |

**Root cause**: The model learns surface-level visual patterns (e.g., "face on paper = famous person") rather than fine-grained distinctions. VizWiz training data may lack sufficient diversity of similar-looking objects with different identities.

### 2. Generic / Vague Descriptions

The model produces overly generic captions that omit key identifying details.

| Image | Prediction | Ground Truth |
|-------|-----------|--------------|
| val_00006521 | "a box of **frozen food** being held" | A box of **Barilla whole grain penne pasta** |
| val_00005001 | "a white sign with black text and a blue logo" | A label showing **item NO:39653, Made in China** |
| val_00006487 | "a **white device** with black and red buttons on carpet" | A **white power strip** with outlets and switches |

**Root cause**: The model defaults to safe, high-frequency category words rather than generating specific descriptions. This is reinforced by VizWiz annotations that sometimes themselves use generic language.

### 3. Hallucination of Specific Details

When the model does attempt specificity, it sometimes generates factually incorrect details.

- Predicting "The Big Bang Theory" for a completely unrelated DVD cover
- Predicting "George Washington" when Lincoln is depicted

**Root cause**: The model memorizes strong co-occurrences from training (DVD → popular shows, money → famous presidents) without grounding in the actual visual content.

### 4. Difficulty with Low-Quality VizWiz Images

VizWiz images are captured by visually impaired users, resulting in frequent blur, poor framing, camera flash glare, and dark/overexposed shots.

| Image | Prediction | Ground Truth |
|-------|-----------|--------------|
| val_00002358 | "a blurry image with a white background and some black spots" | "A brownish object sits around a bright white flash" |
| val_00007401 | "a black screen with nothing on it" | "Image is completely black and there are no items visible" |

**Root cause**: Limited training signal from degraded images. The model hasn't seen enough diverse examples of poorly-lit or blurry scenes to learn meaningful descriptions.

### 5. Inability to Read Text in Images

Labels, screens, and documents are described structurally but without reading the actual text content.

**Root cause**: The frozen ViT-base encoder was not trained for OCR tasks. The 224×224 input resolution further limits text readability. This is an architectural limitation that synthetic data alone cannot fully solve, but exposure to images containing text with descriptive captions could improve structural text awareness.

## Research Question

> **Can targeted synthetic data — generated with Stable Diffusion for underrepresented and commonly confused categories — improve captioning accuracy on VizWiz?**

Specifically, we hypothesize that generating training images for:
- **Commonly confused objects** (currency denominations, bags vs. shoes, specific products)
- **Household electronics** (power strips, remotes, hubs)
- **Scenes with text/labels** (packaging, shipping labels)
- **Low-quality conditions** (dark, blurry, glare)

...and using the generation prompts as ground-truth captions, will:
1. Improve METEOR and BLEU scores on the VizWiz test set
2. Reduce category-level confusion errors
3. Encourage more specific descriptions over generic ones

Both YES and NO outcomes are informative: if synthetic data doesn't help, it reveals that the domain gap between SD3.5 outputs and real VizWiz photos is too large, or that the identified errors stem from architectural limitations rather than data scarcity.

## Target Categories for Synthetic Data

| Category | Weakness Addressed | Example Prompts |
|----------|-------------------|-----------------|
| Currency | Object misidentification | Bills with specific presidents, denominations |
| Product packaging | Generic descriptions | Named brand products, readable labels |
| Electronics | Vague device naming | Power strips, remotes, keyboards with brands |
| Media/DVDs | Hallucinated titles | DVD cases with neutral descriptions |
| Accessories | Object confusion | Bags, shoes, wallets with distinctive features |
| Computer equipment | Missing brand/type info | Specific keyboards, monitors, laptops |
| Text/labels | Text blindness | Shipping labels, nutrition facts |
| Low quality scenes | Poor image handling | Dark rooms, blurry objects, flash glare |
| Fabric/clothing | Generic material descriptions | Specific garment types and colors |
| Food/kitchen | Vague food naming | Specific food items and appliances |
