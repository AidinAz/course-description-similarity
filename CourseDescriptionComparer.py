import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rapidfuzz import fuzz as _fuzz
    _RAPIDFUZZ = True
except ImportError:
    import difflib
    _RAPIDFUZZ = False

_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'can', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
    'what', 'which', 'who', 'how', 'when', 'where', 'all', 'each', 'both',
    'more', 'most', 'other', 'some', 'such', 'not', 'only', 'also',
    'include', 'including', 'course', 'students', 'topic', 'topics', 'key',
    'related', 'part', 'based', 'use', 'using', 'used', 'new', 'well',
    'both', 'their', 'into', 'about', 'between', 'through',
})


def _preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if t not in _STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


def _tfidf_cosine(clean1, clean2):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([clean1, clean2])
    score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    return round(score * 100, 1)


def _keyword_overlap(text1, text2, top_n=25):
    def top_terms(text):
        tokens = _preprocess(text).split()
        unigrams = Counter(tokens)
        bigrams = Counter(f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1))
        return set(dict((unigrams + bigrams).most_common(top_n)))

    kw1, kw2 = top_terms(text1), top_terms(text2)
    union = kw1 | kw2
    jaccard = round(len(kw1 & kw2) / len(union) * 100, 1) if union else 0.0
    return jaccard


def _fuzzy_scores(text1, text2):
    if _RAPIDFUZZ:
        return {
            "token_set_ratio": _fuzz.token_set_ratio(text1, text2),
            "token_sort_ratio": _fuzz.token_sort_ratio(text1, text2),
            "partial_ratio": _fuzz.partial_ratio(text1, text2),
        }
    seq = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
    return {"sequence_ratio": round(seq.ratio() * 100, 1)}


def compare_two_courses_descriptions(first_text, second_text):
    clean1 = _preprocess(first_text)
    clean2 = _preprocess(second_text)

    tfidf_score = _tfidf_cosine(clean1, clean2)
    kw_score = _keyword_overlap(first_text, second_text)
    fuzzy = _fuzzy_scores(first_text, second_text)

    best_fuzzy = max(fuzzy.values())
    composite = round(0.55 * tfidf_score + 0.25 * kw_score + 0.20 * best_fuzzy, 1)

    return {
        "composite_score": composite,
        "tfidf_cosine_similarity": tfidf_score,
        "keyword_overlap_jaccard": kw_score,
        **{f"fuzzy_{k}": v for k, v in fuzzy.items()},
    }


course_first_text = """
First Course Description...
"""

course_second_text = """
Second Course Description...
"""

scores = compare_two_courses_descriptions(course_first_text, course_second_text)

print(f"\n{'=' * 50}")
print("Course Similarity Analysis")
print('=' * 50)
for method, value in scores.items():
    print(f"{method}: {value}")
