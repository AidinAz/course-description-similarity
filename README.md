# Course Description Comparator

A lightweight Python tool that measures semantic similarity between two course descriptions using three complementary NLP methods combined into a single composite score.

---

## How It Works

The comparison pipeline runs three independent analyses and blends them into one score:

| Method | Weight | Description |
|---|---|---|
| TF-IDF Cosine Similarity | 55% | Measures vocabulary overlap weighted by term importance across both texts |
| Keyword Overlap (Jaccard) | 25% | Compares top unigrams and bigrams extracted from each description |
| Fuzzy String Matching | 20% | Catches paraphrasing and reordered content using token-based fuzzy ratios |

**Composite score formula:**
```
composite = 0.55 × tfidf + 0.25 × keyword_jaccard + 0.20 × best_fuzzy
```

All scores are on a **0–100 scale**.

---

## Requirements

```
scikit-learn
rapidfuzz        # optional but recommended — falls back to difflib if missing
```

Install dependencies:

```bash
pip install scikit-learn rapidfuzz
```

---

## Usage

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

---

## Using as a Library

Import the comparison function directly into your own code:

```python
from CourseDescriptionComparer import compare_two_courses_descriptions

result = compare_two_courses_descriptions(desc_a, desc_b)
print(result["composite_score"])  # e.g. 72.4
```

The function returns a dictionary with all individual scores alongside the composite.

---

## Interpreting Scores

| Composite Score | Interpretation |
|---|---|
| 80–100 | Highly similar — likely covering the same material |
| 60–79 | Moderate overlap — shared themes with notable differences |
| 40–59 | Weak overlap — related subject area but distinct focus |
| 0–39 | Low similarity — likely different courses |
