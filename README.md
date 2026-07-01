# Course Description Comparator

A Python toolkit for comparing course descriptions using three complementary approaches: a fast statistical NLP scorer (`CourseStatisticalComparer`), a semantic embedding comparer (`CourseEmbeddingComparer`), and an LLM-based qualitative analyser (`CourseLLMComparer`).

---

## How It Works

### Statistical Comparer (`CourseStatisticalComparer.py`)

Runs three independent NLP analyses and blends them into one composite score:

| Method | Weight | Description |
|---|---|---|
| TF-IDF Cosine Similarity | 40% | Measures vocabulary overlap weighted by term importance across both texts |
| Keyword Overlap (Jaccard) | 20% | Compares top unigrams and bigrams extracted from each description |
| Fuzzy String Matching | 40% | Catches paraphrasing and reordered content using token-based fuzzy ratios |

**Composite score formula:**
```
composite = 0.40 × tfidf + 0.20 × keyword_jaccard + 0.40 × best_fuzzy
```

All scores are on a **0–100 scale**.

### Embedding Comparer (`CourseEmbeddingComparer.py`)

Encodes descriptions with a sentence-transformer model and blends two semantic scores:

| Method | Weight | Description |
|---|---|---|
| Full-text Cosine | 60% | Embeds each full description and computes cosine similarity |
| Sentence Alignment | 40% | Bidirectional best-match similarity across individual sentences |

**Composite score formula:**
```
composite = 0.60 × full_text_cosine + 0.40 × sentence_alignment
```

Sentence alignment is only computed when **both** descriptions have at least 2 detectable sentences. Otherwise it is skipped — `sentence_alignment` is `None`, `sentence_alignment_used` is `False`, and the composite falls back to `full_text_cosine` alone.

All scores are on a **0–100 scale**. Default model: `all-MiniLM-L6-v2` (downloaded once, then cached locally).

#### One-sided "coverage" mode

The default comparison is **symmetric** — it answers *"how similar are these two descriptions?"* That penalises a candidate covering **broader** material, because the extra topics dilute the whole-text similarity (a strict superset can score *lower* than its subset).

Pass `direction="coverage"` to score **one-sidedly** instead — *"are `first_text`'s topics covered by `second_text`?"* For each topic in the first (original) description it asks a **cross-encoder NLI model** whether any topic in the second description *entails* (covers) it, then averages the best per-topic entailment scores. Entailment — rather than plain embedding cosine — is used here because cosine only measures *topical relatedness* and over-counts same-domain-but-distinct topics (e.g. *"Overview of computers and programming"* vs *"Control Structures"*) as covered. Adding more topics to the candidate can only raise this score, so broader coverage is never punished. The cross-encoder model (`cross-encoder/nli-deberta-v3-xsmall`, configurable via `coverage_model`) is downloaded once, then cached locally.

| Field | Meaning |
|---|---|
| `coverage_score` | Directional coverage of `first_text`'s topics by `second_text` (0–100). Becomes the `composite_score` when `direction="coverage"`. |
| `coverage_used` | `True` when both texts yield at least one topic unit, so coverage was actually computed; `False` otherwise (the coverage fields then carry fallback values). |
| `topics_covered` / `topics_total` | How many of the original topics are entailed above `coverage_threshold` (entailment probability, default `0.5`) — e.g. *4 of 6 covered*. |
| `fully_covered` | `True` when **every** original topic is entailed above threshold (`topics_covered == topics_total`) — the direct yes/no for *"does `second_text` fully cover `first_text`?"* `False` whenever any topic is missing. |
| `uncovered_topics` | The list of original topics (from `first_text`) with **no** entailment above threshold in `second_text` — i.e. exactly what the other description is missing. Empty when `fully_covered` is `True`. |

In coverage mode `composite_score = coverage_score` exactly — there is **no** full-text blend, since any blend would re-introduce the dilution it is meant to remove (`full_text_weight` / `sentence_weight` are ignored). The coverage fields are always present in the output, in both modes. Topic splitting understands comma-, semicolon-, newline- and bullet-separated lists and keeps parenthetical lists such as `Program Design (Sequence, Decision & Repetition Structures)` intact.

### LLM Comparer (`CourseLLMComparer.py`)

Sends both descriptions to an OpenAI-compatible LLM and returns a structured qualitative analysis covering:

1. **Shared Topics** — content or skills covered by both courses
2. **Unique to Course 1** — topics present only in the first description
3. **Unique to Course 2** — topics present only in the second description
4. **Overall Assessment** — a qualitative judgment of how similar or different the courses are
5. **Similarity Score** — a numeric score from 0–100 using a defined rubric

---

## Requirements

```
scikit-learn==1.9.0
rapidfuzz==3.14.5              # falls back to difflib if missing
sentence-transformers>=3.0.0   # required for CourseEmbeddingComparer only
openai>=1.0.0                  # required for CourseLLMComparer only
python-dotenv>=1.0.0           # required for CourseLLMComparer only
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Statistical Comparer

Edit the placeholder texts in `CourseStatisticalComparer.py`:

```python
course_first_text = """
Introduction to machine learning: supervised and unsupervised learning,
neural networks, model evaluation, and practical applications.
"""

course_second_text = """
Foundations of ML: covering regression, classification, clustering,
deep learning basics, and hands-on projects.
"""

scores = compare_two_courses_descriptions(course_first_text, course_second_text)
```

Run the script:

```bash
python CourseStatisticalComparer.py
```

**Example output:**

```
==================================================
Course Similarity Analysis
==================================================
composite_score: 76.1
tfidf_cosine_similarity: 68.1
keyword_overlap_jaccard: 74.5
fuzzy_token_set_ratio: 81.0
fuzzy_token_sort_ratio: 79.3
fuzzy_partial_ratio: 85.0
```

### Embedding Comparer

Edit the placeholder texts in `CourseEmbeddingComparer.py` and run:

```bash
python CourseEmbeddingComparer.py
```

**Example output** (the demo runs three reports — coverage of the original by the *other* module, coverage by an *identical* module, and a symmetric run for contrast; `fully_covered` and `uncovered_topics` give the verdict directly):

```
==================================================
Coverage — does the other module cover the original?
==================================================
composite_score: 76.7          # one-sided: two original topics are missing
full_text_cosine: 86.8
sentence_alignment: None
sentence_alignment_used: False
coverage_score: 76.7
coverage_used: True
fully_covered: False           # not every original topic is entailed
topics_covered: 4
topics_total: 6
uncovered_topics: ['Overview of computers and programming', 'overview of C']
direction: coverage
model: all-MiniLM-L6-v2
==================================================
Coverage — identical module (full coverage)
==================================================
composite_score: 100.0
full_text_cosine: 100.0
sentence_alignment: None
sentence_alignment_used: False
coverage_score: 100.0
coverage_used: True
fully_covered: True            # every original topic is entailed
topics_covered: 6
topics_total: 6
uncovered_topics: []
direction: coverage
model: all-MiniLM-L6-v2
==================================================
Symmetric (for contrast only) — original vs other module
==================================================
composite_score: 86.8          # other module penalised by dilution — wrong lens here
full_text_cosine: 86.8
sentence_alignment: None
sentence_alignment_used: False
coverage_score: 76.7
coverage_used: True
fully_covered: False
topics_covered: 4
topics_total: 6
uncovered_topics: ['Overview of computers and programming', 'overview of C']
direction: symmetric
model: all-MiniLM-L6-v2
```

### LLM-Based Comparer

Create a `.env` file in the project root:

```
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://your-llm-endpoint/v1
LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo   # optional, this is the default
```

Edit the placeholder texts in `CourseLLMComparer.py` and run:

```bash
python CourseLLMComparer.py
```

**Example output:**

```
==================================================
LLM Course Comparison Analysis
==================================================
### Shared Topics
- Core machine learning concepts (supervised learning, neural networks)
- Practical/hands-on application

### Unique to Course 1
- Explicit focus on unsupervised learning
- Model evaluation

### Unique to Course 2
- Regression, classification, and clustering as distinct topics
- Deep learning basics as a separate subject area

### Overall Assessment
The courses are moderately similar, sharing a common ML foundation, but Course 2
breaks topics into more granular categories while Course 1 takes a broader survey approach.

### Similarity Score: 65/100
```

---

## Using as a Library

### Statistical Comparer

```python
from CourseStatisticalComparer import compare_two_courses_descriptions

result = compare_two_courses_descriptions(desc_a, desc_b)
print(result["composite_score"])  # e.g. 72.4
```

The function returns a dictionary with all individual scores alongside the composite.

Optional keyword arguments:

```python
result = compare_two_courses_descriptions(
    desc_a, desc_b,
    tfidf_weight=0.40,
    keyword_weight=0.20,
    fuzzy_weight=0.40,    # must sum to 1.0 with the other two weights
)
```

### Embedding Comparer

```python
from CourseEmbeddingComparer import compare_two_courses_descriptions

result = compare_two_courses_descriptions(desc_a, desc_b)
print(result["composite_score"])  # e.g. 74.3
```

Optional keyword arguments:

```python
result = compare_two_courses_descriptions(
    desc_a, desc_b,
    model="all-MiniLM-L6-v2",  # any sentence-transformers model
    full_text_weight=0.60,
    sentence_weight=0.40,        # must sum to 1.0 with full_text_weight
)
```

For **one-sided coverage** — *"are `desc_a`'s topics covered by `desc_b`?"* — pass `direction="coverage"`:

```python
result = compare_two_courses_descriptions(
    desc_a, desc_b,
    direction="coverage",        # composite_score = coverage_score (one-sided)
    coverage_model="cross-encoder/nli-deberta-v3-xsmall",  # any NLI cross-encoder
    coverage_threshold=0.5,      # entailment probability above which a topic counts as "covered"
)
print(result["coverage_score"])                                # e.g. 88.0
print(result["topics_covered"], "of", result["topics_total"])  # e.g. 5 of 6
print(result["fully_covered"])                                 # True only if desc_b covers every desc_a topic
print(result["uncovered_topics"])                              # e.g. ['recursion'] — what desc_b is missing
```

### LLM Comparer

```python
from CourseLLMComparer import compare_courses_with_llm

analysis = compare_courses_with_llm(desc_a, desc_b)
print(analysis)  # qualitative text analysis
```

API credentials can also be passed directly instead of via `.env`:

```python
analysis = compare_courses_with_llm(
    desc_a, desc_b,
    api_key="...",
    base_url="https://your-llm-endpoint/v1",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
)
```

---

## Choosing Between the Three Tools

| | `CourseStatisticalComparer.py` | `CourseEmbeddingComparer.py` | `CourseLLMComparer.py` |
|---|---|---|---|
| **Output** | Numeric score (0–100) | Numeric score (0–100) | Qualitative text analysis |
| **Speed** | Fast (local, no API) | Medium (local; first run downloads model — plus a cross-encoder model in coverage mode) | Slower (external API call) |
| **LLM required** | No | No | Yes |
| **Best for** | Keyword overlap / bulk screening | Semantic similarity | Deep qualitative review |

---

## Interpreting Scores

*(Applies to the statistical and embedding comparers.)*

| Composite Score | Interpretation |
|---|---|
| 80–100 | Highly similar — likely covering the same material |
| 60–79 | Moderate overlap — shared themes with notable differences |
| 40–59 | Weak overlap — related subject area but distinct focus |
| 0–39 | Low similarity — likely different courses |
