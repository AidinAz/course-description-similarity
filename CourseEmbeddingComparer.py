import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_CE_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
_model_cache: dict[str, SentenceTransformer] = {}
_ce_cache: dict[str, CrossEncoder] = {}

_UNIT_MIN_LEN = 3
_COVERAGE_THRESHOLD = 0.5
_OPENERS = "([{"
_CLOSERS = ")]}"
_TOPLEVEL_SEPARATORS = ";,\n\r"
_BULLET_CHARS = "•‣◦●▪·*–—-"


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _get_cross_encoder(model_name: str) -> CrossEncoder:
    if model_name not in _ce_cache:
        _ce_cache[model_name] = CrossEncoder(model_name)
    return _ce_cache[model_name]


def _entailment_index(ce: CrossEncoder) -> int:
    label2id = getattr(ce.model.config, "label2id", None) or {}
    for label, idx in label2id.items():
        if "entail" in label.lower():
            return int(idx)
    return 1


def _clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())


def _embed(texts: list[str], model_name: str) -> np.ndarray:
    return _get_model(model_name).encode(texts, normalize_embeddings=True)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [p.strip() for p in parts if len(p.strip()) >= 10]


def _split_units(text: str) -> list[str]:
    depth = 0
    buf: list[str] = []
    raw_units: list[str] = []
    for ch in text:
        if ch in _OPENERS:
            depth += 1
            buf.append(ch)
        elif ch in _CLOSERS:
            depth = max(0, depth - 1)
            buf.append(ch)
        elif depth == 0 and ch in _TOPLEVEL_SEPARATORS:
            raw_units.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    raw_units.append("".join(buf))

    units: list[str] = []
    for unit in raw_units:
        unit = re.sub(r'\s+', ' ', unit.strip().strip(_BULLET_CHARS).strip())
        if len(unit) >= _UNIT_MIN_LEN:
            units.append(unit)

    if not units:
        whole = _clean(text)
        if whole:
            units = [whole]
    return units


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


def _coverage(
    ref_units: list[str],
    cand_units: list[str],
    ce_model_name: str,
    threshold: float,
) -> tuple[float, int, int, list[str]]:
    n_ref = len(ref_units)
    ce = _get_cross_encoder(ce_model_name)
    ent_idx = _entailment_index(ce)
    pairs = [[cand, ref] for ref in ref_units for cand in cand_units]
    probs = ce.predict(pairs, apply_softmax=True)
    ent = np.asarray(probs)[:, ent_idx].reshape(n_ref, len(cand_units))
    best_per_ref = ent.max(axis=1)
    score = round(float(best_per_ref.mean()) * 100, 1)
    covered_mask = best_per_ref >= threshold
    topics_covered = int(covered_mask.sum())
    uncovered_topics = [ref_units[i] for i in range(n_ref) if not covered_mask[i]]
    return score, topics_covered, n_ref, uncovered_topics


def compare_two_courses_descriptions(
    first_text: str,
    second_text: str,
    *,
    model: str = _DEFAULT_MODEL,
    coverage_model: str = _DEFAULT_CE_MODEL,
    direction: str = "symmetric",
    full_text_weight: float = 0.60,
    sentence_weight: float = 0.40,
    coverage_threshold: float = _COVERAGE_THRESHOLD,
) -> dict[str, float | str | bool | int | list[str] | None]:
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
    if direction not in ("symmetric", "coverage"):
        raise ValueError(
            f"direction must be 'symmetric' or 'coverage'; got {direction!r}"
        )
    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError(
            f"coverage_threshold must be in [0.0, 1.0]; got {coverage_threshold}"
        )

    t1 = _clean(first_text)
    t2 = _clean(second_text)

    full_embs = _embed([t1, t2], model)
    ft_score = _full_text_cosine(full_embs[0], full_embs[1])

    sents1 = _split_sentences(t1)
    sents2 = _split_sentences(t2)
    sentence_alignment_used = len(sents1) >= 2 and len(sents2) >= 2
    sa_score = (
        _sentence_alignment(sents1, sents2, model)
        if sentence_alignment_used
        else None
    )

    ref_units = _split_units(first_text)
    cand_units = _split_units(second_text)
    coverage_used = len(ref_units) >= 1 and len(cand_units) >= 1
    if coverage_used:
        coverage_score, topics_covered, topics_total, uncovered_topics = _coverage(
            ref_units, cand_units, coverage_model, coverage_threshold
        )
    else:
        coverage_score, topics_covered, topics_total = ft_score, 0, len(ref_units)
        uncovered_topics = []
    fully_covered = coverage_used and topics_covered == topics_total

    if direction == "coverage":
        composite = coverage_score
    elif sentence_alignment_used:
        composite = round(full_text_weight * ft_score + sentence_weight * sa_score, 1)
    else:
        composite = ft_score

    return {
        "composite_score": composite,
        "full_text_cosine": ft_score,
        "sentence_alignment": sa_score,
        "sentence_alignment_used": sentence_alignment_used,
        "coverage_score": coverage_score,
        "coverage_used": coverage_used,
        "fully_covered": fully_covered,
        "topics_covered": topics_covered,
        "topics_total": topics_total,
        "uncovered_topics": uncovered_topics,
        "direction": direction,
        "model": model,
    }


if __name__ == "__main__":
    original_module = (
        "Original module description..."
        )

    other_module = (
        "Other module description..."
        )
    identical_module = original_module

    def _report(title: str, second_text: str, **kwargs) -> None:
        scores = compare_two_courses_descriptions(
            original_module, second_text, **kwargs
        )
        print(f"\n{'=' * 50}")
        print(title)
        print('=' * 50)
        for method, value in scores.items():
            print(f"{method}: {value}")

    # Coverage mode answers the real question: does the other module fully
    # cover the original? (Extra topics in the other module are never penalised.)
    _report("Coverage — does the other module cover the original?", other_module,
            direction="coverage")
    _report("Coverage — identical module (full coverage)", identical_module,
            direction="coverage")

    # Symmetric mode is shown only for contrast — it is NOT the right lens here,
    # since it penalises the other module for covering extra material.
    _report("Symmetric (for contrast only) — original vs other module", other_module)
