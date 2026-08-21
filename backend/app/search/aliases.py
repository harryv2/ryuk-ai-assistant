"""Vendor alias expansion.

A person types "Turkish Airlines". Their mailbox says THY, the sender is
`thy.com`, the subject carries TK1984 and the Turkish original spells it
Turk Hava Yollari (in Turkish script). One embedding will not bridge that, and it does not have to:
the mapping is a fact about the world, so it lives in a table
(``aliases.yaml``), is read once, and costs nothing at query time.

Two things come out of a group and both matter downstream:

* **tokens** — every surface form. A token found in a subject or a title is
  ``ALIAS_TOKEN_IN_SUBJECT`` evidence in :mod:`app.search.hybrid`, which forces
  ``cn`` to 1.0 regardless of what the vector arm thought.
* **sender_domains** — an exact ``From:`` domain is ``EXACT_SENDER`` evidence,
  the strongest signal in the system short of an id match.

Matching is on word boundaries over a normalised string (case folded, accents
stripped, punctuation flattened), which is what makes a two-letter token like
``tk`` safe to carry.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).with_name("aliases.yaml")

# Word characters for boundary tests after normalisation: letters, digits and
# the space that joins a multi-word token.
_WORD = re.compile(r"[a-z0-9]+")
_PUNCT = re.compile(r"[^a-z0-9]+")


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def normalise(text: str | None) -> str:
    """Case fold, strip accents, flatten punctuation to single spaces.

    The Turkish-script spelling of Turk Hava Yollari and its ASCII form land on the
    same string, which is the whole point.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    # Turkish dotless ı and İ survive NFKD; fold them by hand before casefold.  # noqa: RUF003
    stripped = stripped.replace("ı", "i").replace("İ", "i").replace("ﬁ", "fi")  # noqa: RUF001
    lowered = stripped.casefold()
    return _PUNCT.sub(" ", lowered).strip()


def tokens_of(text: str | None) -> list[str]:
    """The normalised word list of a string."""
    return _WORD.findall(normalise(text))


def _contains_token(haystack_norm: str, token_norm: str) -> bool:
    """True when ``token_norm`` appears in ``haystack_norm`` on word boundaries."""
    if not token_norm or not haystack_norm:
        return False
    start = 0
    length = len(token_norm)
    while True:
        found = haystack_norm.find(token_norm, start)
        if found < 0:
            return False
        before_ok = found == 0 or haystack_norm[found - 1] == " "
        end = found + length
        after_ok = end == len(haystack_norm) or haystack_norm[end] == " "
        if before_ok and after_ok:
            return True
        start = found + 1


# --------------------------------------------------------------------------- #
# The group
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AliasGroup:
    """One brand and every way it shows up."""

    key: str
    canonical: str
    kind: str = "vendor"
    tokens: tuple[str, ...] = ()
    sender_domains: tuple[str, ...] = ()
    code_patterns: tuple[str, ...] = ()
    #: normalised tokens, longest first, so "turkish airlines" wins over "tk"
    normalised: tuple[str, ...] = field(default=(), repr=False)

    def matches(self, text: str | None) -> bool:
        """True when any surface form appears in ``text``."""
        return bool(self.tokens_in(text))

    def tokens_in(self, text: str | None) -> list[str]:
        """Every surface form of this group present in ``text``, longest first."""
        norm = normalise(text)
        if not norm:
            return []
        return [t for t in self.normalised if _contains_token(norm, t)]

    def matches_sender(self, email: str | None) -> bool:
        """True when an address belongs to one of this group's domains.

        Subdomains count: ``mail.turkishairlines.com`` matches
        ``turkishairlines.com``.
        """
        if not email or "@" not in str(email):
            return False
        domain = str(email).rsplit("@", 1)[-1].strip().lower().rstrip(".")
        return any(
            domain == known or domain.endswith("." + known)
            for known in self.sender_domains
        )

    def codes_in(self, text: str | None) -> list[str]:
        """Every brand code (``TK1984``, an Amazon order number) found in text."""
        if not text:
            return []
        out: list[str] = []
        for pattern in self.compiled_patterns():
            out.extend(m.group(0).strip() for m in pattern.finditer(str(text)))
        seen: set[str] = set()
        unique = []
        for code in out:
            if code.upper() not in seen:
                seen.add(code.upper())
                unique.append(code)
        return unique

    @cache  # noqa: B019 - frozen dataclass, one group per key
    def compiled_patterns(self) -> tuple[re.Pattern[str], ...]:
        compiled = []
        for raw in self.code_patterns:
            try:
                compiled.append(re.compile(raw, re.IGNORECASE))
            except re.error:  # a bad pattern in the table must not break search
                continue
        return tuple(compiled)

    def as_dict(self) -> dict[str, Any]:
        """The shape the pre-pass puts on the intent object."""
        return {
            "alias_group": self.key,
            "canonical": self.canonical,
            "kind": self.kind,
            "tokens": list(self.tokens),
            "sender_domains": list(self.sender_domains),
            "code_patterns": list(self.code_patterns),
        }


# --------------------------------------------------------------------------- #
# Loading — PyYAML when present, a small parser for this file's subset when not
# --------------------------------------------------------------------------- #


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if text[0] in "\"'" and text[-1] == text[0] and len(text) >= 2:
        return text[1:-1]
    if text == "[]":
        return []
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text


def _mini_yaml(source: str) -> dict[str, Any]:
    """Parse the strict subset ``aliases.yaml`` is written in.

    Nested maps by two-space indentation, lists of scalars with ``- ``, inline
    ``[]`` for an empty list, ``#`` comments, single or double quoted scalars.
    Anything else raises, because a silently half-read alias table is worse
    than a loud one.
    """
    root: dict[str, Any] = {}
    # (indent, container) stack. A container is a dict or a list.
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_key: list[tuple[int, str, dict[str, Any]]] = []

    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"aliases.yaml: bad indentation on line {lineno}")

        # A key whose value is on the following lines opens a new container.
        while pending_key and pending_key[-1][0] >= indent:
            pending_key.pop()

        container = stack[-1][1]

        if body.startswith("- "):
            if not isinstance(container, list):
                # The parent key wants a list; create it now.
                if not pending_key:
                    raise ValueError(f"aliases.yaml: stray list item on line {lineno}")
                _, key, parent = pending_key[-1]
                new_list: list[Any] = []
                parent[key] = new_list
                stack.append((indent - 1, new_list))
                container = new_list
            container.append(_parse_scalar(body[2:]))
            continue

        if ":" not in body:
            raise ValueError(f"aliases.yaml: expected 'key: value' on line {lineno}")

        key, _, value = body.partition(":")
        key = key.strip()
        value = value.strip()
        if not isinstance(container, dict):
            raise ValueError(f"aliases.yaml: mapping inside a list on line {lineno}")

        if value:
            container[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            container[key] = child
            stack.append((indent, child))
            pending_key.append((indent, key, container))

    return root


def _load_raw(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return _mini_yaml(source)
    loaded = yaml.safe_load(source) or {}
    if not isinstance(loaded, dict):
        raise ValueError("aliases.yaml must be a mapping")
    return loaded


def _build(raw: dict[str, Any]) -> dict[str, AliasGroup]:
    groups: dict[str, AliasGroup] = {}
    for key, body in (raw.get("groups") or {}).items():
        body = body or {}
        canonical = str(body.get("canonical") or key.replace("_", " ").title())
        seen_forms: set[str] = set()
        forms: list[str] = []
        for candidate in [canonical, *(str(t) for t in (body.get("tokens") or []))]:
            text = candidate.strip()
            # Deduplicate on the literal, not the normalised form: "türk hava
            # yolları" and "turk hava yollari" match the same things but a  # noqa: RUF003
            # person reading the expansion should still see both spellings.
            marker = text.casefold()
            if not text or not normalise(text) or marker in seen_forms:
                continue
            seen_forms.add(marker)
            forms.append(text)
        tokens = tuple(forms)
        normalised = tuple(
            sorted(
                {n for n in (normalise(t) for t in tokens) if n},
                key=lambda s: (-len(s), s),
            )
        )
        groups[key] = AliasGroup(
            key=str(key),
            canonical=canonical,
            kind=str(body.get("kind") or "vendor"),
            tokens=tokens,
            sender_domains=tuple(
                str(d).strip().lower() for d in (body.get("sender_domains") or []) if str(d).strip()
            ),
            code_patterns=tuple(str(p) for p in (body.get("code_patterns") or []) if str(p)),
            normalised=normalised,
        )
    return groups


@lru_cache(maxsize=4)
def _table(path: str) -> tuple[dict[str, AliasGroup], dict[str, str], dict[str, str]]:
    """``(groups, token -> key, domain -> key)``, read once per path."""
    groups = _build(_load_raw(Path(path)))
    by_token: dict[str, str] = {}
    by_domain: dict[str, str] = {}
    for key, group in groups.items():
        for token in group.normalised:
            # First writer wins, so a longer, more specific table entry earlier
            # in the file is not overwritten by a two-letter code later on.
            by_token.setdefault(token, key)
        for domain in group.sender_domains:
            by_domain.setdefault(domain, key)
    return groups, by_token, by_domain


def groups(path: str | Path | None = None) -> dict[str, AliasGroup]:
    """Every alias group, keyed by its group name."""
    return _table(str(path or DEFAULT_PATH))[0]


def reload() -> None:
    """Drop the cached table. For tests and for a config reload."""
    _table.cache_clear()


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #


def find(term: str | None, path: str | Path | None = None) -> AliasGroup | None:
    """The group a term names, or None.

    Tries an exact match on the normalised term first, then looks for any known
    surface form inside it — so both ``"THY"`` and ``"my turkish airlines
    booking"`` resolve to the same group. When several match, the longest
    surface form wins: "air india" beats the bare "ai".
    """
    table, by_token, _ = _table(str(path or DEFAULT_PATH))
    norm = normalise(term)
    if not norm:
        return None
    key = by_token.get(norm)
    if key:
        return table[key]

    best: tuple[int, AliasGroup] | None = None
    for group in table.values():
        for token in group.normalised:
            if _contains_token(norm, token) and (best is None or len(token) > best[0]):
                best = (len(token), group)
                break
    return best[1] if best else None


def find_by_sender(email: str | None, path: str | Path | None = None) -> AliasGroup | None:
    """The group that owns a sender address, or None."""
    table, _, by_domain = _table(str(path or DEFAULT_PATH))
    if not email or "@" not in str(email):
        return None
    domain = str(email).rsplit("@", 1)[-1].strip().lower().rstrip(".")
    key = by_domain.get(domain)
    if key:
        return table[key]
    for known, group_key in by_domain.items():
        if domain.endswith("." + known):
            return table[group_key]
    return None


def expand(term: str | None, path: str | Path | None = None) -> list[str]:
    """Every surface form of ``term``.

    The canonical name first, then the table's tokens, then the term itself if
    the table has never heard of it. Always a list, never empty for a non-empty
    term, so a caller can drop it straight into a query builder.
    """
    text = (term or "").strip()
    group = find(text, path)
    if group is None:
        return [text] if text else []
    out: list[str] = [group.canonical]
    for token in group.tokens:
        if token not in out:
            out.append(token)
    if text and text not in out:
        out.append(text)
    return out


def detect(text: str | None, path: str | Path | None = None) -> list[AliasGroup]:
    """Every group named anywhere in a free-text query, best match first."""
    norm = normalise(text)
    if not norm:
        return []
    scored: list[tuple[int, AliasGroup]] = []
    for group in groups(path).values():
        hits = group.tokens_in(norm)
        if hits:
            scored.append((max(len(h) for h in hits), group))
    scored.sort(key=lambda pair: (-pair[0], pair[1].key))
    return [group for _, group in scored]


def sender_domains(term: str | None, path: str | Path | None = None) -> list[str]:
    """The sender domains for a term, or an empty list."""
    group = find(term, path)
    return list(group.sender_domains) if group else []


def code_patterns(term: str | None, path: str | Path | None = None) -> list[str]:
    """The code regexes for a term, or an empty list."""
    group = find(term, path)
    return list(group.code_patterns) if group else []


def alias_tokens_in(
    text: str | None,
    only: Iterable[str] | Sequence[AliasGroup] | None = None,
    path: str | Path | None = None,
) -> list[str]:
    """Surface forms present in ``text``, optionally restricted to some groups.

    This is what ``hybrid.py`` calls to decide ``ALIAS_TOKEN_IN_SUBJECT``.
    """
    norm = normalise(text)
    if not norm:
        return []
    if only is None:
        chosen: list[AliasGroup] = list(groups(path).values())
    else:
        table = groups(path)
        chosen = []
        for item in only:
            if isinstance(item, AliasGroup):
                chosen.append(item)
            elif isinstance(item, str) and item in table:
                chosen.append(table[item])
            elif isinstance(item, str):
                found = find(item, path)
                if found is not None:
                    chosen.append(found)
    out: list[str] = []
    for group in chosen:
        for token in group.tokens_in(norm):
            if token not in out:
                out.append(token)
    return out


def expansion_for(text: str | None, path: str | Path | None = None) -> dict[str, Any]:
    """The pre-pass block for a query: the first group named, fully expanded.

    Shape matches ``docs/SAMPLE_QUERIES.md`` §4::

        {"alias_group": "turkish_airlines",
         "tokens": [...], "sender_domains": [...], "code_patterns": [...]}

    Returns ``{}`` when no brand is named, which is the common case and is not
    a failure.
    """
    found = detect(text, path)
    if not found:
        return {}
    primary = found[0]
    block = primary.as_dict()
    if len(found) > 1:
        block["also"] = [g.key for g in found[1:]]
    return block


__all__ = [
    "DEFAULT_PATH",
    "AliasGroup",
    "alias_tokens_in",
    "code_patterns",
    "detect",
    "expand",
    "expansion_for",
    "find",
    "find_by_sender",
    "groups",
    "normalise",
    "reload",
    "sender_domains",
    "tokens_of",
]
