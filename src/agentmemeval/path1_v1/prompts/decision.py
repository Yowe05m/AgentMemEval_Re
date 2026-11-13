from __future__ import annotations

import re

from agentmemeval.path1_v1.core.domain import AgentObservation, FactualMemoryRecord, MemoryContext
from agentmemeval.path1_v1.core.poker_position import active_public_players, public_position_label
from agentmemeval.path1_v1.environment.decision_facts import (
    build_decision_facts,
    chips_to_big_blinds,
)
from agentmemeval.path1_v1.environment.raise_sizing import RaiseSizingPlan

PROMPT_TEMPLATE_VERSION = "2026-08-01-paper-en-v2"

BASE_SYSTEM_PROMPT = """
You are a no-limit hold'em decision agent. Maximize long-run expected chip
return while obeying the rules. Use only visible information in the user
message. Compare made hand and draws, position, pot odds, effective stacks, and
public actions before choosing from the listed legal actions.

Numeric chip fields are in chips unless their names end in *_big_blinds. Never
relabel raw chip values as BB. The rule engine's deterministic hand, draw, and
cost analysis overrides your own recognition, preferences, and memory. Do not
claim a pair, two pair, flush, straight, or draw that it rejects.

Memory records past states, actions, and outcomes; it neither proves that a
past action was optimal nor estimates an unselected action. Experience rules
are hypotheses. When a rule matches and no invalidator fires, treat its action
as a preferred candidate, but never let memory override rule-engine facts or
legality.

Output strict JSON:
{"action_type":"fold|check|call|raise","amount":null or integer,
 "confidence":number from 0 to 1,"reason_summary":"short visible-evidence summary",
 "intent":"short strategic objective"}

action_type must be legal. A raise amount is the total raise-to value and must
lie in the supplied closed interval. fold, check, and call require amount=null.
reason_summary must agree with the final action. intent states only the immediate
strategic objective or projected table image, not cards, hand classification,
opponent labels, or a reasoning trace; keep it within 40 characters. A raise near
the maximum usually commits the stack. Never assume access to private cards.
""".strip()


def render_system_prompt(context: MemoryContext) -> str:
    del context
    return BASE_SYSTEM_PROMPT


def render_user_prompt(
    observation: AgentObservation,
    context: MemoryContext,
    raise_sizing: RaiseSizingPlan | None = None,
) -> str:
    # Paper: Eq. (5) assembles visible state, legal actions, and retrieved memory.
    labels = _relative_player_labels(observation)
    lines = [
        "## Current state",
        f"- Street: {observation.phase}",
        f"- Hero seat: {observation.seat}",
        f"- Hero relative position: {public_position_label(observation)}",
        (
            "- Public position anchors: "
            f"BTN seat={observation.dealer_seat}, SB seat={observation.small_blind_seat}, "
            f"BB seat={observation.big_blind_seat}"
        ),
        (
            "- Blind amounts: "
            f"SB={_render_blind_amount(observation.small_blind_amount)}, "
            f"BB={_render_blind_amount(observation.big_blind_amount)}"
        ),
        f"- Hero hole cards: {' '.join(observation.hole_cards)}",
        f"- Board: {' '.join(observation.community_cards) or 'none'}",
        f"- Pot: {_render_chip_amount(observation.pot, observation.big_blind_amount)}",
        (
            "- Current bet: "
            f"{_render_chip_amount(observation.current_bet, observation.big_blind_amount)}"
        ),
        (
            "- Amount to call: "
            f"{_render_chip_amount(observation.to_call, observation.big_blind_amount)}"
        ),
        "",
        "## Deterministic rule-engine analysis (authoritative)",
        *_render_deterministic_analysis(observation),
        "",
        "## Public player state",
    ]
    for player in observation.players:
        if player.busted:
            continue
        lines.append(
            f"- {labels.get(player.agent_id, 'opponent')}: seat={player.seat}, "
            f"stack={_render_chip_amount(player.stack, observation.big_blind_amount)}, "
            f"bet={_render_chip_amount(player.current_bet, observation.big_blind_amount)}, "
            f"folded={player.folded}, all_in={player.all_in}"
        )

    lines.extend(["", "## Legal actions"])
    for action in observation.legal_actions.actions:
        if action.action_type == "raise":
            lines.append(
                "- raise: total raise-to amount in "
                f"[{_render_chip_amount(action.min_amount, observation.big_blind_amount)}, "
                f"{_render_chip_amount(action.max_amount, observation.big_blind_amount)}], "
                f"reopens={action.reopens}"
            )
        else:
            lines.append(f"- {action.action_type}")
    if raise_sizing is not None and raise_sizing.allowed_amounts is not None:
        amounts = ", ".join(
            _render_chip_amount(amount, observation.big_blind_amount)
            for amount in raise_sizing.allowed_amounts
        )
        lines.append(f"- Discrete raise-to candidates: [{amounts}]; a raise must use one of them.")

    lines.extend(["", "## Recent public actions"])
    if observation.action_history:
        for event in observation.action_history[-12:]:
            amount = event.get("amount")
            rendered_amount = (
                _render_chip_amount(amount, observation.big_blind_amount)
                if type(amount) is int
                else ""
            )
            lines.append(
                f"- {event.get('phase')} "
                f"{labels.get(str(event.get('agent_id')), 'opponent')} "
                f"{event.get('action_type')} {rendered_amount}".rstrip()
            )
    else:
        lines.append("- none")
    lines.extend(["", render_memory_section(context)])
    return "\n".join(lines)


def render_facts_section(context: MemoryContext) -> str:
    return _render_facts(context.facts)


def render_experience_section(context: MemoryContext) -> str:
    if context.experience is None:
        return "### Experience document\nnone"
    return f"### Experience document\n{context.experience.body}"


def render_memory_section(context: MemoryContext) -> str:
    sections = [
        "## Memory",
        render_facts_section(context),
        render_experience_section(context),
    ]
    return "\n".join(sections)


def _render_deterministic_analysis(observation: AgentObservation) -> list[str]:
    facts = build_decision_facts(observation)
    draw = facts["draw"]
    hero_draw = draw["hero_draw"]
    board_threat = draw["board_completion_threat"]
    call = facts["call"]
    best_cards = " ".join(facts["best_cards"])
    lines = [f"- Made hand: {facts['made_hand_class']}; best five cards: {best_cards}."]
    if observation.phase == "preflop":
        lines.append(
            "- Draw state: postflop outs are not computed preflop. outs=0 does not imply "
            "that the starting hand has no continuation value."
        )
    elif observation.phase == "river":
        lines.append("- Draw state: the river has no future card; outs=0.")
    else:
        lines.append(
            "- Hero draw: flush_draw={flush_draw}, straight_draw={straight_draw}, "
            "deduplicated nominal outs={outs}, cards_to_come={cards_to_come}. These are "
            "one-card completions involving the hero's hole cards, not necessarily clean "
            "winning outs or equity.".format(
                **hero_draw,
                cards_to_come=draw["cards_to_come"],
            )
        )
        if int(board_threat["outs"]) > 0:
            lines.append(
                "- Board-completion threat: flush={flush_draw}, straight={straight_draw}, "
                "nominal completions={outs}. This shared board change is not a hero draw.".format(
                    **board_threat
                )
            )

    call_cost = int(call["call_cost"])
    if call_cost:
        contestable_pot_before_call = _render_chip_amount(
            int(call["contestable_pot_before_call"]),
            observation.big_blind_amount,
        )
        contestable_pot_after_call = _render_chip_amount(
            int(call["contestable_pot_after_call"]),
            observation.big_blind_amount,
        )
        lines.append(
            "- Call cost: at most "
            f"{_render_chip_amount(call_cost, observation.big_blind_amount)}; "
            "contestable pot before call "
            f"{contestable_pot_before_call}; "
            "contestable pot after call "
            f"{contestable_pot_after_call}; "
            f"required equity approximately {float(call['required_equity']):.1%}; "
            f"fraction of remaining stack {float(call['stack_fraction']):.1%}; "
            f"risk_label={call['risk_label']}, all_in={call['is_all_in']}."
        )
    else:
        lines.append("- Call cost: 0; do not invent a calling cost when checking is available.")

    lines.append(
        f"- Table risk: contestable-pot SPR={float(facts['spr']):.2f}, "
        f"SPR after call={float(facts['spr_after_call']):.2f}; strategic effective stack="
        f"{_render_chip_amount(int(facts['effective_stack']), observation.big_blind_amount)}; "
        f"active players={facts['multiway_players']} (opponents={facts['active_opponents']})."
    )
    raise_rule = observation.legal_actions.rule_for("raise")
    if raise_rule and raise_rule.min_amount is not None and raise_rule.max_amount is not None:
        self_state = next(
            player for player in observation.players if player.agent_id == observation.agent_id
        )
        min_cost = min(self_state.stack, max(0, raise_rule.min_amount - self_state.current_bet))
        max_cost = min(self_state.stack, max(0, raise_rule.max_amount - self_state.current_bet))
        lines.append(
            "- Raise risk: minimum additional investment "
            f"{_render_chip_amount(min_cost, observation.big_blind_amount)}; maximum additional "
            f"investment {_render_chip_amount(max_cost, observation.big_blind_amount)}; maximum "
            f"raise {'is all-in' if max_cost >= self_state.stack else 'is not all-in'}."
        )
    return lines


def _render_blind_amount(amount: int | None) -> str:
    if type(amount) is not int or amount <= 0:
        return "not provided"
    return f"{amount} chips"


def _render_chip_amount(amount: int, big_blind_amount: int | None) -> str:
    rendered = f"{amount} chips"
    big_blinds = chips_to_big_blinds(amount, big_blind_amount)
    if big_blinds is None:
        return rendered
    formatted = f"{big_blinds:.4f}".rstrip("0").rstrip(".")
    return f"{rendered} (={formatted} BB)"


def _render_facts(facts: list[FactualMemoryRecord]) -> str:
    if not facts:
        return "### Factual evidence\nnone"
    lines = [
        "### Factual evidence",
        "These are historical observations, actions, and outcomes, not optimal-action "
        "labels or expected-value estimates.",
    ]
    for fact in facts:
        fact_label = fact.record_id.removeprefix(f"{fact.agent_id}-")
        lines.append(
            f"- {fact_label}: {_sanitize_agent_ids(fact.state_summary, fact.agent_id)}; "
            f"{_sanitize_agent_ids(fact.action_summary, fact.agent_id)}; "
            f"final_reward={fact.final_reward}"
        )
    return "\n".join(lines)


def _relative_player_labels(observation: AgentObservation) -> dict[str, str]:
    labels = {observation.agent_id: "hero"}
    active_players = active_public_players(observation)
    active_seats = [player.seat for player in active_players]
    if observation.seat in active_seats:
        hero_index = active_seats.index(observation.seat)
        active_players = active_players[hero_index + 1 :] + active_players[:hero_index]
    opponents = [player for player in active_players if player.agent_id != observation.agent_id]
    for index, player in enumerate(opponents, start=1):
        labels[player.agent_id] = f"opponent_{index}"
    return labels


_AGENT_ID_PATTERN = re.compile(r"\b(?:agent|heldout)_\d+\b", re.IGNORECASE)


def _sanitize_agent_ids(text: str, owner_id: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return "hero" if match.group(0).lower() == owner_id.lower() else "opponent"

    return _AGENT_ID_PATTERN.sub(replace, str(text))
