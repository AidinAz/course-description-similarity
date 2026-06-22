# Course Description Comparator

A Python toolkit for comparing course descriptions using two complementary approaches: a fast statistical NLP scorer (`CourseDescriptionComparer`) and an LLM-based qualitative analyser (`CourseLLMComparer`).

---

## How It Works

### Statistical Comparer (`CourseDescriptionComparer.py`)

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
rapidfuzz==3.14.5        # falls back to difflib if missing
openai>=1.0.0            # required for CourseLLMComparer only
python-dotenv>=1.0.0     # required for CourseLLMComparer only
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Statistical Comparer

Edit the placeholder texts in `CourseDescriptionComparer.py`:

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
python CourseDescriptionComparer.py
```

**Example output:**

```
==================================================
Course Similarity Analysis
==================================================
composite_score: 72.4
tfidf_cosine_similarity: 68.1
keyword_overlap_jaccard: 74.5
fuzzy_token_set_ratio: 81.0
fuzzy_token_sort_ratio: 79.3
fuzzy_partial_ratio: 85.0
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
from CourseDescriptionComparer import compare_two_courses_descriptions

result = compare_two_courses_descriptions(desc_a, desc_b)
print(result["composite_score"])  # e.g. 72.4
```

The function returns a dictionary with all individual scores alongside the composite.

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

## Choosing Between the Two Tools

| | `CourseDescriptionComparer.py` | `CourseLLMComparer.py` |
|---|---|---|
| **Output** | Numeric score (0–100) | Qualitative text analysis |
| **Speed** | Fast (local, no API) | Slower (external API call) |
| **LLM required** | No | Yes |
| **Best for** | Quick screening / bulk comparison | Deep qualitative review |

---

## Interpreting Scores

*(Applies to the statistical comparer.)*

| Composite Score | Interpretation |
|---|---|
| 80–100 | Highly similar — likely covering the same material |
| 60–79 | Moderate overlap — shared themes with notable differences |
| 40–59 | Weak overlap — related subject area but distinct focus |
| 0–39 | Low similarity — likely different courses |
