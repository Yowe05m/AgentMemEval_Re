"""Conservative deterministic checks for English experience Markdown."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from agentmemeval.path1_v1.prompts.profiles import active_experience_section_labels


@dataclass(frozen=True, slots=True)
class PokerExperienceIssue:
    code: str
    section: str
    bullet: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*-\s+(.+?)\s*$")
_RANK_ORDER = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}
_RANGE_HAND_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<high>[2-9TJQKA])(?P<low>[2-9TJQKA])"
    r"(?P<suffix>[sSoO]?)(?![A-Za-z0-9])"
)
_DEALT_CARD_RE = re.compile(r"(?<![A-Za-z0-9])[2-9TJQKA][shdc](?![A-Za-z0-9])")
_ACTION_RE = re.compile(r"\b(?:fold|check|call|bet|raise|open-raise|jam|all-in)\b", re.I)
_HERO_GUIDANCE_RE = re.compile(
    r"\b(?:hero|we)\b.{0,48}\b(?:should|can|prefer|choose|use|fold|check|call|bet|raise|jam)\b",
    re.I,
)
_PREFLOP_RE = re.compile(r"\b(?:preflop|open-raise|3-?bet|4-?bet|limp|starting hand)\b", re.I)
_FLOP_RE = re.compile(r"\bflop\b", re.I)
_TURN_RE = re.compile(r"\bturn\b", re.I)
_RIVER_RE = re.compile(r"\briver\b", re.I)


def validate_poker_experience_document(body: str) -> list[PokerExperienceIssue]:
    issues: list[PokerExperienceIssue] = []
    for section, bullet in _iter_markdown_bullets(body):
        issues.extend(_validate_bullet(section, bullet))
    return _deduplicate(issues)


def _iter_markdown_bullets(body: str) -> Iterator[tuple[str, str]]:
    section = ""
    for line in body.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            section = heading.group(1).strip()
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            yield section, bullet.group(1).strip()


def _validate_bullet(section: str, bullet: str) -> list[PokerExperienceIssue]:
    checks = (
        _validate_heading,
        _validate_actionability,
        _validate_card_notation,
        _validate_hu_position,
        _validate_action_state,
        _validate_spr,
        _validate_draw_outs,
        _validate_betting_purpose,
        _validate_river_semantics,
        _validate_hand_claims,
        _validate_section_placement,
        _validate_opponent_inference,
    )
    issues: list[PokerExperienceIssue] = []
    for check in checks:
        issues.extend(check(section, bullet))
    return issues


def _validate_heading(section: str, bullet: str) -> list[PokerExperienceIssue]:
    if section in active_experience_section_labels():
        return []
    return [
        _issue("unknown_section", section, bullet, "bullet is outside the five-section contract")
    ]


def _validate_actionability(section: str, bullet: str) -> list[PokerExperienceIssue]:
    issues: list[PokerExperienceIssue] = []
    if not _ACTION_RE.search(bullet):
        issues.append(
            _issue("missing_action", section, bullet, "rule does not name a concrete poker action")
        )
    elif not _HERO_GUIDANCE_RE.search(bullet):
        issues.append(
            _issue(
                "missing_hero_guidance",
                section,
                bullet,
                "rule must distinguish the hero recommendation from opponent actions",
            )
        )
    if re.search(r"\b(?:always|never|guaranteed|certainly)\b", bullet, re.I) and not re.search(
        r"\b(?:if|when|unless|except)\b", bullet, re.I
    ):
        issues.append(
            _issue(
                "unconditional_rule",
                section,
                bullet,
                "absolute prescription lacks a visible condition",
            )
        )
    return issues


def _validate_card_notation(section: str, bullet: str) -> list[PokerExperienceIssue]:
    issues: list[PokerExperienceIssue] = []
    if _DEALT_CARD_RE.search(bullet):
        issues.append(
            _issue(
                "concrete_suited_card",
                section,
                bullet,
                "use ranges or hand classes, not dealt suits",
            )
        )
    for match in _RANGE_HAND_RE.finditer(bullet):
        high, low, suffix = match.group("high", "low", "suffix")
        if high == low and suffix:
            issues.append(
                _issue(
                    "pair_suffix",
                    section,
                    bullet,
                    "pocket pairs must not use suited/offsuit suffixes",
                )
            )
        if _RANK_ORDER[high] < _RANK_ORDER[low]:
            issues.append(
                _issue(
                    "hand_notation_order", section, bullet, "write range notation high rank first"
                )
            )
        if (
            re.search(r"\bconnected\b", bullet, re.I)
            and abs(_RANK_ORDER[high] - _RANK_ORDER[low]) != 1
        ):
            issues.append(
                _issue("connectivity_label", section, bullet, "connected ranks must be adjacent")
            )
    return issues


def _validate_hu_position(section: str, bullet: str) -> list[PokerExperienceIssue]:
    if not re.search(r"\b(?:heads-up|HU)\b", bullet, re.I):
        return []
    issues: list[PokerExperienceIssue] = []
    if re.search(r"BTN.{0,35}(?:faces|calls).{0,12}BB.{0,8}limp", bullet, re.I):
        issues.append(
            _issue(
                "hu_preflop_order",
                section,
                bullet,
                "BTN/SB acts before BB preflop and cannot first face a BB limp",
            )
        )
    if re.search(r"BTN.{0,24}(?:in position|acts last)", bullet, re.I) and not re.search(
        r"\b(?:flop|turn|river|postflop)\b", bullet, re.I
    ):
        issues.append(
            _issue(
                "position_street_scope",
                section,
                bullet,
                "heads-up positional claims must name the street",
            )
        )
    if re.search(r"\b(?:isolate|isolation|get heads-up)\b", bullet, re.I):
        issues.append(
            _issue(
                "hu_isolation",
                section,
                bullet,
                "isolation is undefined when only one opponent exists",
            )
        )
    if (
        re.search(r"BTN.{0,30}check(?:s)?\b", bullet, re.I)
        and re.search(r"\bpostflop\b", bullet, re.I)
        and not re.search(r"(?:after|when).{0,24}BB.{0,12}check", bullet, re.I)
    ):
        issues.append(
            _issue(
                "hu_check_back_order",
                section,
                bullet,
                "BTN can check back only after BB checks postflop",
            )
        )
    return issues


def _validate_action_state(section: str, bullet: str) -> list[PokerExperienceIssue]:
    issues: list[PokerExperienceIssue] = []
    if re.search(r"(?:facing|against).{0,28}(?:bet|raise)", bullet, re.I) and re.search(
        r"\bhero\s+(?:should|can|may)\s+check\b", bullet, re.I
    ):
        issues.append(
            _issue(
                "check_facing_bet",
                section,
                bullet,
                "check is illegal while facing an outstanding bet",
            )
        )
    if (
        re.search(r"\bhero\s+(?:should|can|may)\s+raise\b", bullet, re.I)
        and not re.search(r"(?:facing|after|against).{0,30}(?:bet|raise)", bullet, re.I)
        and not _PREFLOP_RE.search(bullet)
    ):
        issues.append(
            _issue(
                "raise_without_bet",
                section,
                bullet,
                "a postflop raise requires an outstanding bet or raise",
            )
        )
    if re.search(r"(?:opponent|villain).{0,18}re-?raises?", bullet, re.I) and not re.search(
        r"hero.{0,28}raise", bullet, re.I
    ):
        issues.append(
            _issue(
                "reraise_chain",
                section,
                bullet,
                "an opponent re-raise requires an intervening hero raise",
            )
        )
    return issues


def _validate_spr(section: str, bullet: str) -> list[PokerExperienceIssue]:
    if not re.search(r"\bSPR\b", bullet):
        return []
    if _PREFLOP_RE.search(bullet) and not re.search(
        r"\b(?:projected|target|flop)\s+SPR\b", bullet, re.I
    ):
        return [
            _issue(
                "preflop_spr",
                section,
                bullet,
                "preflop references must say projected or target flop SPR",
            )
        ]
    if re.search(
        r"\bdeep[- ]stacked\b.{0,24}\b(?:means|equals|is)\s+(?:a\s+)?high\s+SPR", bullet, re.I
    ):
        return [
            _issue(
                "stack_spr_conflation",
                section,
                bullet,
                "deep stacks and current postflop SPR are not equivalent",
            )
        ]
    return []


def _validate_draw_outs(section: str, bullet: str) -> list[PokerExperienceIssue]:
    issues: list[PokerExperienceIssue] = []
    gutshot = re.search(r"\b(?:gutshot|inside straight draw)\b.{0,20}(\d+)\s+outs?", bullet, re.I)
    if gutshot and int(gutshot.group(1)) != 4:
        issues.append(
            _issue("gutshot_outs", section, bullet, "a clean gutshot has four nominal outs")
        )
    open_ended = re.search(r"\b(?:open[- ]ended|OESD)\b.{0,20}(\d+)\s+outs?", bullet, re.I)
    if open_ended and int(open_ended.group(1)) != 8:
        issues.append(
            _issue(
                "open_ended_outs", section, bullet, "a clean open-ended draw has eight nominal outs"
            )
        )
    return issues


def _validate_betting_purpose(section: str, bullet: str) -> list[PokerExperienceIssue]:
    issues: list[PokerExperienceIssue] = []
    if re.search(
        r"\b(?:bet|raise).{0,28}(?:for information|to find out|pot control)\b", bullet, re.I
    ):
        issues.append(
            _issue(
                "invalid_bet_purpose",
                section,
                bullet,
                "betting purpose must be value, bluff, pricing, or equity denial",
            )
        )
    if re.search(r"\bcall.{0,28}(?:for protection|deny equity)\b", bullet, re.I):
        issues.append(
            _issue(
                "call_protection", section, bullet, "calling does not deny the opponent's equity"
            )
        )
    if re.search(r"\bvalue (?:bet|raise)\b", bullet, re.I) and not re.search(
        r"\b(?:worse|weaker).{0,20}(?:call|continue)\b", bullet, re.I
    ):
        issues.append(
            _issue(
                "value_premise_missing",
                section,
                bullet,
                "value betting must identify worse hands that continue",
            )
        )
    return issues


def _validate_river_semantics(section: str, bullet: str) -> list[PokerExperienceIssue]:
    if not _RIVER_RE.search(bullet):
        return []
    issues: list[PokerExperienceIssue] = []
    if re.search(r"\b(?:pot control|protect|free card|draw|outs?)\b", bullet, re.I):
        issues.append(
            _issue(
                "river_future_card",
                section,
                bullet,
                "river decisions have no future-card protection, outs, or pot-control branch",
            )
        )
    return issues


def _validate_hand_claims(section: str, bullet: str) -> list[PokerExperienceIssue]:
    issues: list[PokerExperienceIssue] = []
    if re.search(
        r"paired board.{0,35}(?:gives|means|is).{0,18}(?:two pair|trips|full house)", bullet, re.I
    ):
        issues.append(
            _issue(
                "paired_board_claim",
                section,
                bullet,
                "a paired board alone does not determine the hero's made hand",
            )
        )
    if re.search(
        r"(?:three|four)[- ]flush board.{0,30}(?:gives|means|is).{0,12}flush", bullet, re.I
    ):
        issues.append(
            _issue(
                "board_flush_claim",
                section,
                bullet,
                "a flush-heavy board alone does not give the hero a flush",
            )
        )
    if re.search(
        r"(?:four[- ]straight|connected) board.{0,30}(?:gives|means|is).{0,12}straight",
        bullet,
        re.I,
    ):
        issues.append(
            _issue(
                "board_straight_claim",
                section,
                bullet,
                "a connected board alone does not give the hero a straight",
            )
        )
    return issues


def _validate_section_placement(section: str, bullet: str) -> list[PokerExperienceIssue]:
    streets = recommended_action_streets(bullet)
    starting_hands, postflop_play, turn_and_river, _opponent, _sizing = (
        active_experience_section_labels()
    )
    if "preflop" in streets and section != starting_hands:
        return [
            _issue(
                "preflop_section", section, bullet, "preflop prescriptions belong in Starting Hands"
            )
        ]
    if streets and streets <= {"turn", "river"} and section != turn_and_river:
        return [
            _issue(
                "late_street_section",
                section,
                bullet,
                "turn- or river-only prescriptions belong in Turn and River",
            )
        ]
    if "flop" in streets and section == turn_and_river:
        return [
            _issue(
                "flop_section",
                section,
                bullet,
                f"flop prescriptions belong in {postflop_play}",
            )
        ]
    return []


def _validate_opponent_inference(section: str, bullet: str) -> list[PokerExperienceIssue]:
    if re.search(
        r"(?:after|because of) (?:one|a single|this) (?:check|call|fold|raise).{0,35}"
        r"(?:is|must be|proves).{0,12}(?:nit|station|aggressive|passive|bluffer)",
        bullet,
        re.I,
    ):
        return [
            _issue(
                "single_action_archetype",
                section,
                bullet,
                "one public action cannot establish an opponent archetype",
            )
        ]
    if re.search(r"opponent (?:has|holds|must have) [2-9TJQKA]{2}[so]?\b", bullet, re.I):
        return [
            _issue(
                "hidden_hand_inference", section, bullet, "do not assert an opponent's hidden hand"
            )
        ]
    return []


def recommended_action_streets(bullet: str) -> frozenset[str]:
    streets: set[str] = set()
    if _PREFLOP_RE.search(bullet):
        streets.add("preflop")
    if _FLOP_RE.search(bullet):
        streets.add("flop")
    if _TURN_RE.search(bullet):
        streets.add("turn")
    if _RIVER_RE.search(bullet):
        streets.add("river")
    if _ACTION_RE.search(bullet) and not streets:
        streets.add("unspecified")
    return frozenset(streets)


def _issue(code: str, section: str, bullet: str, message: str) -> PokerExperienceIssue:
    return PokerExperienceIssue(code=code, section=section, bullet=bullet, message=message)


def _deduplicate(issues: list[PokerExperienceIssue]) -> list[PokerExperienceIssue]:
    seen: set[tuple[str, str, str]] = set()
    result: list[PokerExperienceIssue] = []
    for issue in issues:
        key = (issue.code, issue.section, issue.bullet)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


__all__ = [
    "PokerExperienceIssue",
    "recommended_action_streets",
    "validate_poker_experience_document",
]
