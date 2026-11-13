"""Prompts for revising the five-section poker experience document."""

EXPERIENCE_TITLE = "# My Experience"
EXPERIENCE_SECTION_HEADINGS = (
    "Starting Hands",
    "Postflop Play",
    "Turn and River",
    "Opponent Adaptation",
    "Bet Sizing and Position",
)


EXPERIENCE_UPDATE_PROMPT = """
Revise the long-term poker experience document only when the supplied evidence
changes a transferable decision rule. Preserve valid rules and edit only the
affected sections, but return the complete Markdown document when keep=false.

The document must begin with '# My Experience' and contain exactly these five
sections in order: Starting Hands, Postflop Play, Turn and River, Opponent
Adaptation, and Bet Sizing and Position. Each non-empty section contains
actionable rules with an observable condition, a preferred legal action or
sizing principle, and any observable condition that would change that choice.
Do not write hand histories, recent-return summaries, sample counts, or generic
disclaimers.

The document must generalize to unfamiliar tables. Do not include player IDs,
hand IDs, fact IDs, seat numbers, or hidden information. Opponent adaptations
must refer to reusable archetypes or public action lines. A trajectory proves
only its recorded state, committed action, and outcome; it does not reveal the
counterfactual value of actions not taken. Objective card, stack, public-action,
committed-action, and deterministic-decision-fact fields override the model's
declared intent whenever they conflict.

Compare the decision-time intent with the observed net reward, public actions,
and visible showdown in calibration_note. In self_check, identify conflicts
between the candidate rules and supplied facts. supporting_fact_ids may contain
only IDs present in this request and must be empty when no facts are supplied.

Output one JSON object with exactly keep, new_md, calibration_note, self_check,
and supporting_fact_ids. With keep=true, new_md is empty or exactly equals the
current document. With keep=false, new_md is the complete five-section Markdown
document. Do not output patches, analysis, or Markdown fences.
""".strip()


EXPERIENCE_MAIN_REFLECTION_PROMPT = """
You have just completed a hand of no-limit hold'em. Update your knowledge only
when the hand supports a transferable judgment. Separate decision quality from
card variance: winning does not validate an action and losing does not refute
one. No update is often the correct result. Output only the JSON requested by
the user prompt.
""".strip()


EXPERIENCE_MAIN_EXP_REVISE_INSTRUCTION = """
Use the current hand review, recent-hand recap, and current experience document
to decide whether a revision is warranted. Preserve the five-section structure
and prefer a small edit over a rewrite.

The experience document is long-term and table-independent. Never identify a
specific player, rotating seat label, seat number, or hand. In Opponent
Adaptation, describe only an archetype or reusable public action line.

Calibration: compare the recorded decision-time intent with the committed
action and observed outcome. Note material expectation-outcome mismatches, but
do not infer the value of an action that was not selected.

Self-check: test every candidate rule against the supplied facts. Explain any
resolved conflict in self_check; otherwise use 'no conflict found'.

Output exactly:
{
  "keep": true,
  "new_md": "complete five-section Markdown when keep is false",
  "calibration_note": "intent versus observed outcome",
  "self_check": "consistency with supplied facts",
  "supporting_fact_ids": []
}
""".strip()


EXPERIENCE_PAPER_OBJECTIVE_COMPATIBILITY_ADDENDUM = """
Objective records include hole cards, board cards, public actions, committed
actions, outcomes, and deterministic decision facts. declared_intent_not_ground_truth
is a model self-report, not evidence of cards, hand class, or action quality.

The complete document is limited to 64 rules and 16 rules per section. Capacity
is a ceiling, not a target. Consolidate duplicates before adding a rule. If the
document is empty but the recap contains real decisions, create a small initial
playbook from repeated decision problems and conservative poker priors; mark
prior-only claims appropriately and do not present them as outcome evidence.

Each bullet should independently guide one decision: observable condition ->
one preferred currently legal action or sizing -> purpose. Add an exception only
when it changes the action or sizing. Starting-hand examples are limited to two
per rule; otherwise use a range or hand class. Static action order and legality
belong to the engine, not the experience document.

With keep=false, return the complete Markdown document beginning with
'# My Experience'. The program validates positions, terminology, hand classes,
action chains, section placement, duplicates, capacity, and evidence boundaries.
Validation failure keeps the previous document unchanged.
""".strip()


EXPERIENCE_PAPER_STRATEGY_REVIEW_CARD = """
Before writing a rule, verify its poker semantics:

- A bet or raise must derive value from worse calls, better folds, equity denial,
  or pricing. A raise requires an outstanding opponent bet or raise. Calls and
  folds must reference price, required equity, range, or effective stack.
- A check must specify the current action state and a later branch. Frequent
  checking, a single win, or intent-outcome agreement does not make checking
  optimal. Strong hands do not automatically require pot control.
- One check, call, or fold cannot define an opponent archetype. Repeated public
  actions are required; distinguish check-call from check-fold tendencies.
- Heads-up, the button is also the small blind: BTN/SB acts first preflop and
  second postflop; BB acts second preflop and first postflop. Completing the
  small blind is not calling an opponent limp.
- Behavior-digest warnings describe coverage, not strategy. Low proactive-action
  coverage must not turn frequent checks into a default rule.
""".strip()


EXPERIENCE_PAPER_UPDATE_PROMPT = EXPERIENCE_MAIN_EXP_REVISE_INSTRUCTION


EXPERIENCE_MAIN_ASYNC_SWEEP_INSTRUCTION = """
Revise the experience document from (1) retrieved facts grouped by retrieval
path, (2) hands since the previous sweep, and (3) the current document. Preserve
the five-section structure and make the smallest justified edit.

Leave a fact unclassified by default. Mark a fact contradicting only when its
observable record directly conflicts with a current rule; mark it noise only
when it is a non-transferable one-off outcome or obvious error; mark facts
supporting only when multiple facts support one consolidated transferable rule.
Generalize player-specific observations to an archetype or public action line.

Output one JSON object containing keep, new_md, calibration_note, self_check,
supporting_fact_ids, contradicting_fact_ids, and noise_fact_ids. All fact-ID
arrays must be pairwise disjoint and may reference only IDs in this request.
""".strip()


EXPERIENCE_PAPER_ASYNC_FACT_CLASSIFICATION_PROMPT = EXPERIENCE_MAIN_ASYNC_SWEEP_INSTRUCTION


EXPERIENCE_ASYNC_FACT_CLASSIFICATION_PROMPT = """
This is a FactExprAsync sweep. Return exactly the seven fields required by the
asynchronous instruction. supporting_fact_ids, contradicting_fact_ids, and
noise_fact_ids must be pairwise disjoint subsets of the supplied fact IDs.
Omitted facts remain unclassified.
""".strip()


EXPERIENCE_TRANSFERABLE_QUALITY_PROMPT = """
Apply the transferable_rulebook_v2 quality gate. The document is a compact
long-term playbook, not a hand-example archive: at most 16 rules per section
and 64 in total. Merge synonymous or subsumed rules across all five sections;
replace a lower-value rule when new evidence materially corrects the document.

Every rule must name an observable state and a preferred hero action or sizing.
Do not present an opponent action as the recommendation, offer an unresolved
menu, or use vague advice such as 'be careful'. A rule must stand alone outside
the source hand and should include a visible condition that reverses it when
such a condition is material.

Use range notation such as AKs, KQo, 22+, or Axs, never dealt cards with suits.
The suffix s means suited and o means offsuit; pocket pairs take no suffix.
Write ranks high-to-low and call hands connected only when ranks are adjacent.

Heads-up, BTN is SB. BTN/SB acts first preflop and second postflop; BB acts
second preflop and first postflop. State the street for positional claims.
Check-back is available only after the opponent checks. Do not recommend
isolation when only one opponent exists.

A paired board does not automatically give the hero two pair, trips, or a full
house. A three- or four-flush board does not itself give the hero a flush, and a
connected board does not itself give the hero a straight. Distinguish board
texture, hero hand, and hero draw. Use the fixed hand-class order: straight
flush, four of a kind, full house, flush, straight, three of a kind, two pair,
one pair, high card.

SPR means postflop effective stack divided by pot; preflop references must say
projected or target flop SPR. A gutshot normally has four theoretical outs and
an open-ended draw eight, subject to dirty outs. Checking through the flop can
reveal only the turn; checking through the turn can reveal only the river.
Do not describe betting as information gathering or pot control. On the river,
future-card protection and pot-control rationales are inapplicable.

Place starting-hand and preflop range prescriptions in Starting Hands; flop
rules in Postflop Play; turn- or river-only rules in Turn and River; archetype
adjustments in Opponent Adaptation; and general stack, sizing, action-order, and
position principles in Bet Sizing and Position. Separate decision quality from
variance and do not update when there is no new observable information.
""".strip()


EXPERIENCE_ASYNC_TRANSFERABLE_FACT_CLASSIFICATION_PROMPT = (
    EXPERIENCE_ASYNC_FACT_CLASSIFICATION_PROMPT
    + " Classify facts only when the evidence is explicit. Consolidate multiple "
    "supporting facts into one rule; never add one rule per fact."
)


EXPERIENCE_PATCH_UPDATE_PROMPT = """
Maintain the long-term playbook with the smallest structured patch. Do not
rewrite Markdown. The renderer owns the five sections: Starting Hands,
Postflop Play, Turn and River, Opponent Adaptation, and Bet Sizing and Position.

operation is keep or patch. keep requires patches=[]. Every patch has exactly
section, op, rule_id, when, guidance, invalidators, basis, status,
supporting_trajectory_ids, and supporting_fact_ids. add leaves rule_id empty;
revise and delete reference an existing rule. add/revise require one to four
observable conditions and invalidators. guidance names an action and sizing.
delete leaves all content and evidence arrays empty.

basis=observed_evidence may cite only supplied evidence. basis=model_prior must
use status=tentative and empty evidence IDs. Trajectories are observations, not
optimal-action labels, and do not reveal counterfactual outcomes. Do not infer
that an observed fold, call, check, or raise dominates unobserved actions.

Do not include identities, hand IDs, seat numbers, sample counts, average
returns, evidence IDs, or hidden information in rule text. fact_assessments may
use only supplied fact IDs and the labels supporting, contradicting, noise, or
irrelevant. Output only schema-conforming JSON and no reasoning trace.
""".strip()
