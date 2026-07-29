#!/usr/bin/env python3
"""Audit generated question rewrites and materialize a safe public-only panel.

The natural-paraphrase generator is intentionally left unchanged so its raw
output remains an immutable diagnostic artifact.  This script applies a
deterministic, label-blind equivalence gate to each proposed rewrite.  A failed
rewrite is replaced by a wrapper that contains the original question verbatim;
wrappers are explicitly marked and are not natural-paraphrase evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from one_confidence_frozen_source_access import load_rows, require_same_access


SCHEMA_VERSION = "one-confidence-question-equivalence-audit-v1"
RULESET_VERSION = "target-anchor-constraint-bidirectional-lexical-v1"
PUBLIC_SPLITS = ("development", "validation")

# These wrappers change only the answer-format instruction.  Each contains the
# source question byte-for-byte, making the factual request deterministic.
FALLBACK_WRAPPERS = (
    "Answer this unchanged question directly: {question}",
    "Give only the shortest factual answer to this unchanged question: {question}",
    "Identify the requested entity or value without explanation. Unchanged question: {question}",
    "Respond with only the answer to this unchanged question: {question}",
)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
MONTH_CANONICAL = {
    "jan": "january",
    "january": "january",
    "feb": "february",
    "february": "february",
    "mar": "march",
    "march": "march",
    "apr": "april",
    "april": "april",
    "may": "may",
    "jun": "june",
    "june": "june",
    "jul": "july",
    "july": "july",
    "aug": "august",
    "august": "august",
    "sep": "september",
    "sept": "september",
    "september": "september",
    "oct": "october",
    "october": "october",
    "nov": "november",
    "november": "november",
    "dec": "december",
    "december": "december",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "could",
    "did", "do", "does", "for", "from", "give", "had", "has", "have", "here", "identify",
    "in", "into", "is", "it", "its", "name", "of", "on", "only", "or", "please", "provide",
    "question", "requested", "state", "tell", "that", "the", "their", "there", "these", "this",
    "those", "to", "was", "were", "what", "when", "where", "which", "who", "whom", "whose",
    "will", "with", "would", "you", "your",
}

# Only high-confidence lexical equivalences are included.  This mapping is used
# for overlap diagnostics, never to infer an answer or a correctness label.
SYNONYMS = {
    "authored": "write", "author": "write", "wrote": "write", "written": "write", "writer": "write",
    "composed": "compose", "composer": "compose", "composition": "compose",
    "created": "create", "creator": "create", "made": "create",
    "country": "country", "nation": "country",
    "count": "count", "number": "count", "total": "count",
    "located": "location", "location": "location", "place": "location",
    "inaugural": "first", "initial": "first",
    "final": "last",
    "present-day": "current", "present": "current",
    "previous": "former",
    "biggest": "largest", "greatest": "largest",
    "smallest": "smallest", "lowest": "lowest", "highest": "highest",
}

TARGET_NOUNS = {
    "country": "country", "nation": "country",
    "city": "settlement", "town": "settlement", "village": "settlement", "capital": "settlement",
    "state": "administrative_area", "province": "administrative_area", "county": "administrative_area",
    "continent": "continent",
    "river": "geographic_feature", "mountain": "geographic_feature", "sea": "geographic_feature",
    "ocean": "geographic_feature", "island": "geographic_feature", "lake": "geographic_feature",
    "person": "person", "individual": "person", "man": "person", "woman": "person", "actor": "person", "actress": "person",
    "author": "person", "writer": "person", "composer": "person", "director": "person",
    "musician": "person", "artist": "person", "singer": "person", "performer": "person", "hero": "person",
    "legend": "person", "star": "person", "mathematician": "person", "politician": "person",
    "comedian": "person", "comedienne": "person", "inventor": "person", "explorer": "person",
    "minister": "person", "chairman": "person", "president": "person", "king": "person", "queen": "person", "scientist": "person",
    "organization": "organization", "organisation": "organization", "company": "organization",
    "team": "organization", "band": "organization", "group": "organization",
    "station": "facility", "plant": "facility", "airport": "facility", "building": "facility", "stadium": "facility",
    "language": "language", "religion": "religion", "currency": "currency",
    "song": "creative_work", "film": "creative_work", "movie": "creative_work", "book": "creative_work", "piece": "creative_work",
    "novel": "creative_work", "opera": "creative_work", "album": "creative_work", "show": "creative_work", "series": "creative_work",
    "title": "title", "name": "name",
    "location": "location", "place": "location", "site": "location",
    "sport": "category", "genre": "category", "category": "category", "type": "category",
    "occupation": "category", "profession": "category", "style": "category",
    "religion": "category", "tradition": "category", "material": "material", "substance": "material",
}

QUESTION_LEADS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "name", "identify",
    "give", "state", "tell", "provide", "is", "are", "was", "were", "do", "does", "did", "has",
    "have", "had", "will", "would", "can", "could", "should",
}

PROPER_EXCLUSIONS = {
    "A", "An", "And", "Are", "As", "At", "By", "Can", "Could", "Did", "Do", "Does", "For",
    "From", "Give", "Has", "Have", "How", "Identify", "In", "Is", "Name", "Of", "On", "Please",
    "State", "Tell", "The", "To", "Was", "Were", "What", "When", "Where", "Which", "Who", "Whom",
    "Whose", "Why", "Will", "Would", "His", "Her", "Its", "Their", "This", "These", "Those",
}

ACRONYM_EXPANSIONS = {
    "uk": ("united kingdom", "britain", "british"),
    "us": ("united states", "america", "american"),
    "usa": ("united states", "america", "american"),
    "ussr": ("soviet union",),
    "uae": ("united arab emirates",),
    "un": ("united nations",),
    "eu": ("european union",),
}

CONSTRAINT_PATTERNS = {
    "negation": re.compile(r"\b(?:not|never|neither|without|except|excluding)\b"),
    "first": re.compile(r"\b(?:first|inaugural|initial)\b"),
    "last": re.compile(r"\b(?:last|final)\b"),
    "current": re.compile(r"\b(?:current|currently|present-day)\b"),
    "former": re.compile(r"\b(?:former|previous|ex-)\b"),
    "before": re.compile(r"\bbefore\b"),
    "after": re.compile(r"\bafter\b"),
    "earliest": re.compile(r"\bearliest\b"),
    "latest": re.compile(r"\blatest\b"),
    "oldest": re.compile(r"\boldest\b"),
    "youngest": re.compile(r"\byoungest\b"),
    "largest": re.compile(r"\b(?:largest|biggest|greatest)\b"),
    "smallest": re.compile(r"\bsmallest\b"),
    "highest": re.compile(r"\bhighest\b"),
    "lowest": re.compile(r"\blowest\b"),
    "most": re.compile(r"\bmost\b"),
    "least": re.compile(r"\bleast\b"),
    "only": re.compile(r"\bonly\b"),
}

SEMANTIC_AXIS_PATTERNS = {
    "origin": re.compile(r"\b(?:origin|originate|originated|emerge|emerged|come from|came from)\b"),
    "material": re.compile(r"\b(?:material|made of|made from|constructed of|constructed from|composed of)\b"),
    "price": re.compile(r"\b(?:price|cost|sold for|purchased for|bought for)\b"),
    "cause": re.compile(r"\b(?:why|reason|caused by|because of|due to)\b"),
    "birth": re.compile(r"\b(?:born|birth|birthplace)\b"),
    "death": re.compile(r"\b(?:died|death|dead|passed away|killed)\b"),
    "capital_relation": re.compile(r"\bcapital\b"),
    "occupation": re.compile(r"\b(?:occupation|profession|worked as|job)\b"),
    "classification": re.compile(r"\b(?:genre|category|type of|kind of|classify|classification)\b"),
    "authorship": re.compile(r"\b(?:author|authored|write|writes|wrote|written|compose|composed|composer)\b"),
    "production": re.compile(r"\b(?:producer|produced|producing)\b"),
    "performance": re.compile(r"\b(?:performed|performer|sang|sung|singer|duet|duetted|portrayed|played the (?:role|part)|starred)\b"),
}


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def simple_stem(token: str) -> str:
    token = SYNONYMS.get(token, token)
    if len(token) > 5 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return SYNONYMS.get(token, token)


def content_tokens(text: str) -> set[str]:
    output = set()
    for token in normalize(text).split():
        if token in STOPWORDS or token in NUMBER_WORDS or token.isdigit():
            continue
        output.add(simple_stem(token))
    return output


def numeric_constraints(text: str) -> set[str]:
    values = set()
    for token in normalize(text).split():
        if token.isdigit():
            values.add(token)
        elif re.fullmatch(r"\d{2,4}s", token):
            values.add(token)
        elif token in NUMBER_WORDS:
            values.add(NUMBER_WORDS[token])
    return values


def quoted_anchors(text: str) -> list[str]:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"'):
        stripped = stripped[1:-1]
    collapsed = re.sub(r'"{2,}', '"', stripped)
    spans = re.findall(r'"([^"\n]{2,160})"', collapsed)
    spans += re.findall(r"“([^”\n]{2,160})”", text)
    spans += re.findall(r"‘([^’\n]{2,160})’", text)
    spans += re.findall(r"(?<![A-Za-z])'([^'\n]{2,160})'(?![A-Za-z])", text)
    output = set()
    for span in spans:
        value = normalize(span)
        if value and len(value.split()) <= 16:
            output.add(value)
    return sorted(output)


def proper_anchors(text: str) -> list[str]:
    tokens = WORD_RE.findall(text)
    anchors = []
    for index, token in enumerate(tokens):
        plain = re.sub(r"['’]s$", "", token.replace("’", "'"), flags=re.IGNORECASE).strip("'")
        if plain in PROPER_EXCLUSIONS:
            continue
        has_internal_upper = any(char.isupper() for char in plain[1:])
        is_acronym = len(plain) >= 2 and plain.replace("-", "").isupper()
        is_proper = plain[:1].isupper()
        if is_acronym or has_internal_upper or is_proper:
            key = normalize(plain)
            if key and key not in MONTH_CANONICAL and key not in STOPWORDS:
                anchors.append(key)
    return sorted(set(anchors))


def anchor_present(anchor: str, candidate: str) -> bool:
    candidate_normalized = normalize(candidate)
    candidate_tokens = set(candidate_normalized.split())
    if set(anchor.split()) <= candidate_tokens:
        return True
    return any(expansion in candidate_normalized for expansion in ACRONYM_EXPANSIONS.get(anchor, ()))


def constraint_signature(text: str) -> set[str]:
    normalized = normalize(text)
    return {name for name, pattern in CONSTRAINT_PATTERNS.items() if pattern.search(normalized)}


def temporal_anchors(text: str) -> set[str]:
    return {MONTH_CANONICAL[token] for token in normalize(text).split() if token in MONTH_CANONICAL}


def semantic_axis_signature(text: str) -> set[str]:
    normalized = normalize(text)
    return {name for name, pattern in SEMANTIC_AXIS_PATTERNS.items() if pattern.search(normalized)}


def _target_noun(text: str) -> str | None:
    normalized = normalize(text)
    if re.match(r"^what is .+ the capital of\b", normalized):
        return "entity"
    if re.search(r"\bsovereign state\b", normalized):
        return "country"
    if re.search(r"\b(?:capital city|administrative cent(?:er|re))\b", normalized):
        return "settlement"
    if re.search(r"\b(?:film|movie|television|tv) (?:star|director|actor|actress)\b", normalized):
        return "person"

    # Restrict noun search to the interrogative/request head. Looking through
    # arbitrary context caused false classifications such as treating the opera
    # in "Which musician composed the opera ...?" as the answer type.
    matches = list(re.finditer(r"\b(?:what|which)\b", normalized))
    for match in reversed(matches):
        tail = normalized[match.end() :].strip().split()[:7]
        if not tail:
            continue
        named_prefix = len(tail) >= 3 and tail[0] in {"is", "was", "are", "were"} and tail[1:3] == ["the", "name"]
        if named_prefix or tail[:2] == ["the", "name"]:
            for token in tail[3:] if named_prefix else tail[2:]:
                if token in TARGET_NOUNS and token != "name":
                    return TARGET_NOUNS[token]
            return "name"
        if tail[:2] in (["is", "called"], ["was", "called"]):
            return "name"
        for token in tail:
            if token in {"does", "did", "has", "have", "had"}:
                break
            if token in TARGET_NOUNS:
                return TARGET_NOUNS[token]

    imperative = re.search(r"\b(?:name|identify|give|state|provide)\b(.{0,90})", normalized)
    if imperative:
        tail = imperative.group(1).split()[:10]
        if tail[:2] == ["the", "year"] or (tail and tail[0] == "year"):
            return "date"
        for token in tail:
            if token in TARGET_NOUNS:
                return TARGET_NOUNS[token]
    return None


def compatible_targets(source: str, rewrite: str) -> bool:
    if source == rewrite:
        return True
    # Generic naming forms preserve the requested answer type when all factual
    # anchors and relation content pass the independent checks below.
    generic = {"entity", "name", "title"}
    if source in generic and rewrite in generic:
        return True
    if {source, rewrite} <= {"creative_work", "name", "title"}:
        return True
    if {source, rewrite} <= {"count", "quantity"}:
        return True
    return False


def answer_target(text: str) -> str:
    normalized = normalize(text)
    tokens = normalized.split()
    prefix = " ".join(tokens[:8])
    polite_request = bool(
        re.match(r"^(?:can|could|would|will) you (?:name|identify|tell|give|state|provide)\b", normalized)
    )
    # A polar lead takes precedence over nouns such as "number" in the body:
    # "Is X bordered by a number of countries?" asks yes/no, not for a count.
    if (
        tokens
        and tokens[0] in {"is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "will", "would", "can", "could", "should"}
        and not polite_request
    ):
        return "binary"
    if re.search(r"\bhow many\b|\b(?:number|count|total) of\b|\bexact number\b", normalized):
        return "count"
    if re.search(r"\bhow long\b|\bduration\b", prefix):
        return "duration"
    if re.search(r"\bhow old\b|\bage of\b", prefix):
        return "age"
    if re.search(r"\bhow far\b|\bdistance\b", prefix):
        return "distance"
    if re.search(r"\bwhat percentage\b|\bwhat proportion\b|\bhow much\b", normalized):
        return "quantity"
    if re.search(r"\b(?:what|which)(?: is| was)? (?:the )?(?:price|cost|material|substance)\b", normalized):
        return "material" if re.search(r"\b(?:material|substance)\b", prefix) else "quantity"
    if re.search(
        r"\b(?:what|which)(?: is| was)? (?:the )?(?:(?:calendar|fiscal) )?(?:year|date|day|month|century|decade|period)\b"
        r"|\bname (?:the )?year\b|^when\b",
        normalized,
    ):
        return "date"
    if (
        tokens
        and tokens[0] in {"who", "whom", "whose"}
        or re.search(r"^(?:with|to|by|from) whom\b", normalized)
        or re.search(r"\b(?:of|with|by|from|to) who(?:m)?\s*$", normalized)
    ):
        return "person"
    if re.match(r"^how (?:would|do|can|should) you classify\b", normalized):
        return "category"
    noun = _target_noun(text)
    if noun is not None:
        return noun
    if tokens and tokens[0] == "where":
        return "location"
    if tokens and tokens[0] == "why":
        return "reason"
    if tokens and tokens[0] == "how":
        return "manner"
    # Polite forms such as "Can you name ..." are requests, not polar questions.
    if polite_request:
        return noun or "entity"
    if tokens and tokens[0] in {"name", "identify", "give", "state", "tell", "provide", "what", "which"}:
        return noun or "entity"
    return "unknown"


def is_question_like(text: str) -> bool:
    tokens = normalize(text).split()
    return bool(tokens) and (text.rstrip().endswith("?") or tokens[0] in QUESTION_LEADS)


def audit_rewrite(source_question: str, proposed_rewrite: str) -> dict[str, Any]:
    """Return a deterministic, label-blind audit decision for one rewrite."""

    reasons: list[str] = []
    source_target = answer_target(source_question)
    rewrite_target = answer_target(proposed_rewrite)
    source_numbers = numeric_constraints(source_question)
    rewrite_numbers = numeric_constraints(proposed_rewrite)
    source_temporal = temporal_anchors(source_question)
    rewrite_temporal = temporal_anchors(proposed_rewrite)
    source_quotes = quoted_anchors(source_question)
    source_anchors = proper_anchors(source_question)
    missing_quotes = [anchor for anchor in source_quotes if not anchor_present(anchor, proposed_rewrite)]
    missing_anchors = [anchor for anchor in source_anchors if not anchor_present(anchor, proposed_rewrite)]
    source_constraints = constraint_signature(source_question)
    rewrite_constraints = constraint_signature(proposed_rewrite)
    source_axes = semantic_axis_signature(source_question)
    rewrite_axes = semantic_axis_signature(proposed_rewrite)
    source_content = content_tokens(source_question)
    rewrite_content = content_tokens(proposed_rewrite)
    shared_content = source_content & rewrite_content
    source_recall = len(shared_content) / max(1, len(source_content))
    rewrite_precision = len(shared_content) / max(1, len(rewrite_content))

    if not proposed_rewrite.strip():
        reasons.append("empty_rewrite")
    if normalize(proposed_rewrite) == normalize(source_question):
        reasons.append("not_a_rewrite")
    if not is_question_like(proposed_rewrite):
        reasons.append("not_question_like")
    if proposed_rewrite.count("?") > 1:
        reasons.append("multiple_questions")
    if re.search(r"\b(?:the answer is|answer:|correct answer)\b", proposed_rewrite, re.IGNORECASE):
        reasons.append("answer_revealing_form")
    if source_target != "unknown" and not compatible_targets(source_target, rewrite_target):
        reasons.append(f"answer_target_mismatch:{source_target}->{rewrite_target}")

    missing_numbers = sorted(source_numbers - rewrite_numbers)
    added_numbers = sorted(rewrite_numbers - source_numbers)
    if missing_numbers:
        reasons.append("missing_numeric_constraints:" + ",".join(missing_numbers))
    if added_numbers:
        reasons.append("added_numeric_constraints:" + ",".join(added_numbers))
    missing_temporal = sorted(source_temporal - rewrite_temporal)
    added_temporal = sorted(rewrite_temporal - source_temporal)
    if missing_temporal:
        reasons.append("missing_temporal_anchors:" + ",".join(missing_temporal))
    if added_temporal:
        reasons.append("added_temporal_anchors:" + ",".join(added_temporal))
    if missing_quotes:
        reasons.append("missing_quoted_anchors:" + ",".join(missing_quotes))

    # Requiring at least 75% of explicit names/acronyms is deliberately
    # conservative.  A rejected rewrite is retained in the audit and replaced,
    # never used to remove the benchmark sample.
    anchor_coverage = 1.0 - len(missing_anchors) / max(1, len(source_anchors))
    if source_anchors and anchor_coverage < 0.75:
        reasons.append("missing_named_anchors:" + ",".join(missing_anchors))

    missing_constraints = sorted(source_constraints - rewrite_constraints)
    added_constraints = sorted(rewrite_constraints - source_constraints)
    if missing_constraints:
        reasons.append("missing_constraint_operators:" + ",".join(missing_constraints))
    if added_constraints:
        reasons.append("added_constraint_operators:" + ",".join(added_constraints))
    missing_axes = sorted(source_axes - rewrite_axes)
    added_axes = sorted(rewrite_axes - source_axes)
    if missing_axes:
        reasons.append("missing_semantic_axes:" + ",".join(missing_axes))
    if added_axes:
        reasons.append("added_semantic_axes:" + ",".join(added_axes))

    # Short questions have too few content words for ratios to be stable.  For
    # longer questions, check coverage in both directions to detect both dropped
    # relations and newly introduced factual clauses.
    if len(source_content) >= 4 and source_recall < 0.38:
        reasons.append(f"low_source_content_recall:{source_recall:.3f}")
    if len(rewrite_content) >= 4 and rewrite_precision < 0.32:
        reasons.append(f"low_rewrite_content_precision:{rewrite_precision:.3f}")

    return {
        "accepted": not reasons,
        "status": "accepted_model_rewrite" if not reasons else "rejected_model_rewrite",
        "reasons": reasons or ["all_deterministic_checks_passed"],
        "diagnostics": {
            "source_answer_target": source_target,
            "rewrite_answer_target": rewrite_target,
            "source_numeric_constraints": sorted(source_numbers),
            "rewrite_numeric_constraints": sorted(rewrite_numbers),
            "source_temporal_anchors": sorted(source_temporal),
            "rewrite_temporal_anchors": sorted(rewrite_temporal),
            "source_quoted_anchors": source_quotes,
            "missing_quoted_anchors": missing_quotes,
            "source_named_anchors": source_anchors,
            "missing_named_anchors": missing_anchors,
            "named_anchor_coverage": anchor_coverage,
            "source_constraint_operators": sorted(source_constraints),
            "rewrite_constraint_operators": sorted(rewrite_constraints),
            "source_semantic_axes": sorted(source_axes),
            "rewrite_semantic_axes": sorted(rewrite_axes),
            "source_content_recall": source_recall,
            "rewrite_content_precision": rewrite_precision,
            "shared_content_tokens": sorted(shared_content),
        },
    }


def fallback_text(question: str, index: int) -> str:
    if not 0 <= index < len(FALLBACK_WRAPPERS):
        raise IndexError(index)
    return FALLBACK_WRAPPERS[index].format(question=question)


def compile_audit(
    benchmark: dict[str, Any],
    features: dict[str, Any],
    cell: str,
    *,
    benchmark_rows: list[tuple[str, dict[str, Any]]] | None = None,
    locked_accessed: bool = False,
    authorization_id: str | None = None,
) -> dict[str, Any]:
    if not locked_accessed and features.get("locked_accessed") is not False:
        raise ValueError("feature artifact must explicitly declare locked_accessed=false")
    require_same_access(
        features,
        frozen=locked_accessed,
        authorization_id=authorization_id,
        source_name="natural paraphrase features",
    )
    if features.get("cell") != cell:
        raise ValueError("feature artifact cell does not match --cell")
    if cell not in benchmark.get("cells", {}):
        raise KeyError(f"unknown benchmark cell: {cell}")

    if benchmark_rows is None:
        benchmark_rows = [
            (split, row)
            for split in PUBLIC_SPLITS
            for row in benchmark["cells"][cell][split]["records"]
        ]
    public_rows = {row["sample_id"]: row for _, row in benchmark_rows}
    allowed_splits = {split for split, _ in benchmark_rows}
    feature_rows = features.get("records", [])
    feature_ids = [row["sample_id"] for row in feature_rows]
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError("duplicate sample_id in feature artifact")
    if set(feature_ids) != set(public_rows):
        if locked_accessed:
            raise ValueError("feature artifact must cover exactly the authorized frozen rows")
        raise ValueError("feature artifact must cover exactly the public development/validation rows")
    if any(row.get("split") not in allowed_splits for row in feature_rows):
        if locked_accessed:
            raise ValueError("feature artifact contains an unauthorized split")
        raise ValueError("feature artifact contains a non-public split")

    audited_records = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    accepted_histogram: Counter[int] = Counter()
    for feature_row in feature_rows:
        sample_id = feature_row["sample_id"]
        benchmark_row = public_rows[sample_id]
        if feature_row["split"] != benchmark_row["split"]:
            raise ValueError(f"split mismatch for {sample_id}")
        proposed = feature_row.get("paraphrases")
        if not isinstance(proposed, list) or len(proposed) != 4:
            raise ValueError(f"expected exactly four proposed rewrites for {sample_id}")

        question = benchmark_row["question"]
        rewrites = []
        selected_panel = []
        seen_selected = set()
        for index, raw_rewrite in enumerate(proposed):
            decision = audit_rewrite(question, raw_rewrite)
            selected = raw_rewrite if decision["accepted"] else fallback_text(question, index)
            selected_source = "model_rewrite" if decision["accepted"] else "deterministic_wrapper_fallback"
            selected_key = normalize(selected)
            if selected_key in seen_selected:
                # Duplicate handling is deterministic and is recorded as another
                # rejection reason rather than silently shrinking the panel.
                decision["accepted"] = False
                decision["status"] = "rejected_model_rewrite"
                if "duplicate_selected_panel_text" not in decision["reasons"]:
                    decision["reasons"].append("duplicate_selected_panel_text")
                selected = fallback_text(question, index)
                selected_source = "deterministic_wrapper_fallback"
                selected_key = normalize(selected)
            seen_selected.add(selected_key)
            selected_panel.append(selected)
            status_counts[decision["status"]] += 1
            for reason in decision["reasons"]:
                reason_counts[reason.split(":", 1)[0]] += 1
            rewrites.append(
                {
                    "rewrite_index": index,
                    "raw_text": raw_rewrite,
                    "raw_text_sha256": text_sha256(raw_rewrite),
                    "audit_status": decision["status"],
                    "audit_reasons": decision["reasons"],
                    "audit_diagnostics": decision["diagnostics"],
                    "selected_text": selected,
                    "selected_text_sha256": text_sha256(selected),
                    "selected_source": selected_source,
                    "eligible_as_natural_paraphrase": bool(decision["accepted"]),
                    "fallback_contains_source_verbatim": question in selected if not decision["accepted"] else None,
                }
            )

        accepted_count = sum(row["eligible_as_natural_paraphrase"] for row in rewrites)
        accepted_histogram[accepted_count] += 1
        audited_records.append(
            {
                "sample_id": sample_id,
                "split": feature_row["split"],
                "source_question": question,
                "source_question_sha256": text_sha256(question),
                "rewrite_raw_output": feature_row.get("rewrite_raw_output", ""),
                "rewrite_raw_output_sha256": text_sha256(feature_row.get("rewrite_raw_output", "")),
                "original_rewrite_status": feature_row.get("rewrite_status"),
                "rewrites": rewrites,
                "equivalent_mask": [rewrite["eligible_as_natural_paraphrase"] for rewrite in rewrites],
                "review_status": "deterministic_rule_audited_v1",
                "selected_panel": selected_panel,
                "panel_questions": [question, *selected_panel],
                "panel_question_sha256": [text_sha256(value) for value in [question, *selected_panel]],
                "selected_panel_sha256": stable_sha256(selected_panel),
                "accepted_model_rewrite_count": accepted_count,
                "fallback_count": 4 - accepted_count,
                "eligible_for_natural_recovery_analysis": accepted_count >= 2,
            }
        )

    corrected_panel_records = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "source_question": row["source_question"],
            "source_question_sha256": row["source_question_sha256"],
            "panel_questions": row["panel_questions"],
            "panel_question_sha256": row["panel_question_sha256"],
            "panel_sha256": stable_sha256(row["panel_questions"]),
            "natural_paraphrase_mask": [
                True,
                *[rewrite["eligible_as_natural_paraphrase"] for rewrite in row["rewrites"]],
            ],
            "accepted_model_rewrite_count": row["accepted_model_rewrite_count"],
            "fallback_count": row["fallback_count"],
            "eligible_for_natural_recovery_analysis": row["eligible_for_natural_recovery_analysis"],
        }
        for row in audited_records
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "ruleset_version": RULESET_VERSION,
        "cell": cell,
        "locked_accessed": locked_accessed,
        "source_authorization_id": authorization_id,
        "decision_is_label_blind": True,
        "decision_inputs": ["source_question", "proposed_rewrite"],
        "fallback_policy": {
            "templates": list(FALLBACK_WRAPPERS),
            "source_question_included_verbatim": True,
            "fallback_is_natural_paraphrase_evidence": False,
        },
        "natural_recovery_eligibility": "at least two accepted model rewrites; fallback wrappers never count",
        "summary": {
            "records": len(audited_records),
            "proposed_rewrites": 4 * len(audited_records),
            "status_counts": dict(sorted(status_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "accepted_model_rewrites_per_record": {
                str(key): accepted_histogram[key] for key in sorted(accepted_histogram)
            },
            "records_eligible_for_natural_recovery_analysis": sum(
                row["eligible_for_natural_recovery_analysis"] for row in audited_records
            ),
        },
        "corrected_panel_manifest": {
            "schema_version": "one-confidence-corrected-question-panel-v1",
            "cell": cell,
            "locked_accessed": locked_accessed,
            "source_authorization_id": authorization_id,
            "contains_labels": False,
            "panel_order": ["canonical", "rewrite_1", "rewrite_2", "rewrite_3", "rewrite_4"],
            "records": corrected_panel_records,
            "records_sha256": stable_sha256(corrected_panel_records),
        },
        "records": audited_records,
    }


def standalone_panel_payload(audit_payload: dict[str, Any]) -> dict[str, Any]:
    """Build the label-free artifact consumed by corrected-panel GPU runs."""

    panel = dict(audit_payload["corrected_panel_manifest"])
    panel["ruleset_version"] = audit_payload["ruleset_version"]
    panel["audit_artifact_sha256"] = audit_payload["artifact_sha256_excluding_this_field"]
    panel["input_provenance"] = audit_payload["input_provenance"]
    panel["artifact_sha256_excluding_this_field"] = stable_sha256(panel)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-output", type=Path)
    parser.add_argument("--frozen-source-authorization", type=Path)
    args = parser.parse_args()

    benchmark, benchmark_rows, locked_accessed, authorization_id = load_rows(
        benchmark_path=args.benchmark,
        cell=args.cell,
        authorization_path=args.frozen_source_authorization,
        stage="question_equivalence_audit",
        output_path=args.output,
    )
    if locked_accessed and args.panel_output is not None:
        raise ValueError("frozen audit embeds the panel and forbids --panel-output")
    benchmark_raw = args.benchmark.read_bytes()
    feature_raw = args.features.read_bytes()
    features = json.loads(feature_raw)
    payload = compile_audit(
        benchmark,
        features,
        args.cell,
        benchmark_rows=benchmark_rows,
        locked_accessed=locked_accessed,
        authorization_id=authorization_id,
    )
    payload["input_provenance"] = {
        "benchmark_path": str(args.benchmark),
        "benchmark_sha256": hashlib.sha256(benchmark_raw).hexdigest(),
        "features_path": str(args.features),
        "features_sha256": hashlib.sha256(feature_raw).hexdigest(),
    }
    payload["artifact_sha256_excluding_this_field"] = stable_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.panel_output is not None:
        panel_payload = standalone_panel_payload(payload)
        args.panel_output.parent.mkdir(parents=True, exist_ok=True)
        args.panel_output.write_text(
            json.dumps(panel_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
