import re
import numpy as np
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())


def _embed(texts: list[str], model_name: str) -> np.ndarray:
    return _get_model(model_name).encode(texts, normalize_embeddings=True)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [p.strip() for p in parts if len(p.strip()) >= 10]


def _full_text_cosine(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return round(float(max(0.0, np.dot(emb1, emb2))) * 100, 1)


def _sentence_alignment(sents1: list[str], sents2: list[str], model_name: str) -> float:
    all_embs = _embed(sents1 + sents2, model_name)
    emb1 = all_embs[:len(sents1)]
    emb2 = all_embs[len(sents1):]
    sim_matrix = emb1 @ emb2.T
    forward = float(sim_matrix.max(axis=1).mean())
    backward = float(sim_matrix.max(axis=0).mean())
    return round((forward + backward) / 2 * 100, 1)


def compare_two_courses_descriptions(
    first_text: str,
    second_text: str,
    *,
    model: str = _DEFAULT_MODEL,
    full_text_weight: float = 0.60,
    sentence_weight: float = 0.40,
) -> dict[str, float | str | bool]:
    if not isinstance(first_text, str) or not isinstance(second_text, str):
        raise TypeError(
            "Both arguments must be strings; "
            f"got {type(first_text).__name__!r} and {type(second_text).__name__!r}"
        )
    if not first_text.strip() or not second_text.strip():
        raise ValueError("Both course descriptions must be non-empty.")
    if abs(full_text_weight + sentence_weight - 1.0) > 0.001:
        raise ValueError(
            f"Weights must sum to 1.0; got {full_text_weight + sentence_weight}"
        )

    t1 = _clean(first_text)
    t2 = _clean(second_text)

    full_embs = _embed([t1, t2], model)
    ft_score = _full_text_cosine(full_embs[0], full_embs[1])

    sents1 = _split_sentences(t1)
    sents2 = _split_sentences(t2)

    sentence_alignment_used = len(sents1) >= 2 and len(sents2) >= 2
    sa_score = _sentence_alignment(sents1, sents2, model) if sentence_alignment_used else ft_score

    composite = round(full_text_weight * ft_score + sentence_weight * sa_score, 1)

    return {
        "composite_score": composite,
        "full_text_cosine": ft_score,
        "sentence_alignment": sa_score,
        "sentence_alignment_used": sentence_alignment_used,
        "model": model,
    }


if __name__ == "__main__":
    course_first_text = """
    First course description...
    """

    course_second_text = """
    Second course description...
    """

    scores = compare_two_courses_descriptions(course_first_text, course_second_text)

    print(f"\n{'=' * 50}")
    print("Embedding-Based Course Similarity")
    print('=' * 50)
    for method, value in scores.items():
        print(f"{method}: {value}")
