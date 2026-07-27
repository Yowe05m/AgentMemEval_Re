"""Deterministic, conservative poker semantics checks for experience Markdown.

The validator deliberately scans only Markdown bullets.  It reports high-confidence
contradictions and does not try to grade strategy quality or infer hidden game state.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PokerExperienceIssue:
    """One deterministic semantic issue tied to its source Markdown bullet."""

    code: str
    section: str
    bullet: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation with a stable field order."""

        return asdict(self)


_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*-\s+(.+?)\s*$")
_SUFFIXED_HAND_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<high>[2-9TJQKA])(?P<low>[2-9TJQKA])"
    r"(?P<suffix>[sSoO])(?![A-Za-z0-9])",
)
_RANKED_HAND_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<high>[2-9TJQKA])(?P<low>[2-9TJQKA])"
    r"(?P<suffix>[sSoO]?)(?![A-Za-z0-9])(?!(?:\s*)[%％])",
)
_HYPHENATED_HOLE_CARD_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<first>[2-9TJQKA])\s*[-－—–]\s*"
    r"(?P<second>[2-9TJQKA])\s*(?P<suffix>同花|非同花|不同花|杂色|s|o)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_NOTATION_META_NEGATION_RE = re.compile(
    r"(?:不要|不应|不能|不可|避免).{0,16}(?:写成|写作|书写为|记作|表述为)\s*$"
)
_OFFSUIT_TERM_RE = re.compile(
    r"(?:非同花|不同花|杂色|花色不同|花色不一致|异色)"
    r"(?!的?(?:牌面|公共牌|公牌)|面|结构)|off[- ]?suit",
    re.IGNORECASE,
)
_SUITED_TERM_RE = re.compile(
    r"(?<!非)(?<!不)同花(?!牌面|听牌|结构|完成|可能|潜力|公共牌)|"
    r"(?:花色相同|花色一致|同色)(?!的?(?:牌面|公共牌|公牌)|牌面|听牌|结构)|"
    r"suited",
    re.IGNORECASE,
)
_COLLECTIVE_SUIT_ASSERTION_RE = re.compile(
    r"都是|均为|皆为|同为|都属于|均属于|皆属于|both\s+are|all\s+are",
    re.IGNORECASE,
)
_DIRECT_LABEL_ASSERTION_RE = re.compile(
    r"(?:是|属于|作为|称为|视为|算作)|\b(?:is|are|called)\b",
    re.IGNORECASE,
)
_COMPARISON_LABEL_RE = re.compile(
    r"比较|相较|相比|版本|compare|versus|\bvs\.?\b",
    re.IGNORECASE,
)
_HAND_LABEL_BINDING_BARRIER_RE = re.compile(
    r"[、）)]|(?:或者|或|以及|及|和|与)|\b(?:or|and)\b",
    re.IGNORECASE,
)
_CONNECTOR_TERM_RE = re.compile(r"连张|连接牌|connectors?", re.IGNORECASE)
_CONNECTIVITY_NEGATION_RE = re.compile(
    r"(?:无|没有|缺乏|不具备).{0,4}(?:连张|连接牌|connectors?)",
    re.IGNORECASE,
)
_GAPPER_TERM_RE = re.compile(
    r"隔张|间隔牌|一间隔|两间隔|gappers?",
    re.IGNORECASE,
)
_CLAUSE_BREAK_RE = re.compile(r"[。；;\n]")

_RANK_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

_POSTFLOP_RE = re.compile(
    r"翻后|翻牌后|看翻牌|转牌|河牌|post[- ]?flop|\bflop\b|\bturn\b|\briver\b",
    re.IGNORECASE,
)
_EXPLICIT_POSTFLOP_RE = re.compile(
    r"翻后|翻牌后|翻牌圈|转牌|河牌|post[- ]?flop|\bflop\b|\bturn\b|\briver\b",
    re.IGNORECASE,
)
_EXPLICIT_PREFLOP_RE = re.compile(r"翻牌前|\bpre[- ]?flop\b", re.IGNORECASE)
_BTN_RE = re.compile(r"(?<![A-Za-z0-9])BTN(?![A-Za-z0-9])|按钮位|庄家位", re.IGNORECASE)
_BB_RE = re.compile(r"(?<![A-Za-z0-9])BB(?![A-Za-z0-9])|大盲位?|big blind", re.IGNORECASE)
_SB_RE = re.compile(r"(?<![A-Za-z0-9])SB(?![A-Za-z0-9])|小盲位?|small blind", re.IGNORECASE)
_HU_COMBINED_BUTTON_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:BTN\s*/\s*SB|SB\s*/\s*BTN)(?![A-Za-z0-9])|"
    r"(?:BTN|按钮位|庄家位)\s*(?:同时(?:是|为)?|兼任|即)\s*(?:SB|小盲位?)|"
    r"(?:SB|小盲位?)\s*(?:同时(?:是|为)?|兼任|即)\s*(?:BTN|按钮位|庄家位)",
    re.IGNORECASE,
)
_POSITION_POLARITY_RE = re.compile(
    r"有利位置|不利位置|有位置|无位置|位置内|位置外|位置优势|位置劣势|"
    r"先行动|率先行动|首先行动|后行动|最后行动|\bIP\b|\bOOP\b",
    re.IGNORECASE,
)
_POSITION_POLARITY_NEGATION_RE = re.compile(
    r"(?:并?无|毫无|没有|不具备|不具有|不存在)\s*"
    r"(?:有利|不利)?位置(?:优势|劣势)?",
    re.IGNORECASE,
)
_BTN_WRONG_RE = re.compile(
    r"先行动|率先行动|首先行动|不利位置|无位置|位置外|\bOOP\b|位置劣势",
    re.IGNORECASE,
)
_BB_WRONG_RE = re.compile(
    r"后行动|最后行动|有利位置|有位置|位置内|\bIP\b|位置优势",
    re.IGNORECASE,
)
_BTN_PREFLOP_WRONG_RE = re.compile(
    r"后行动|最后行动|有利位置|有位置|位置内|\bIP\b|位置优势",
    re.IGNORECASE,
)
_BB_PREFLOP_WRONG_RE = re.compile(
    r"先行动|率先行动|首先行动|不利位置|无位置|位置外|\bOOP\b|位置劣势",
    re.IGNORECASE,
)
_TWO_PLAYER_TABLE_RE = re.compile(
    r"\bHU\b|两人桌|双人桌|单挑(?:牌局|桌|游戏)|"
    r"heads?[- ]?up\s+(?:table|game)",
    re.IGNORECASE,
)
_BB_VS_SB_RE = re.compile(
    r"(?:BB|大盲位?|big blind).{0,16}(?:对|面对|相对|对抗).{0,6}"
    r"(?:SB|小盲位?|small blind)|"
    r"(?:对|面对|相对|对抗).{0,6}(?:SB|小盲位?|small blind).{0,16}"
    r"(?:BB|大盲位?|big blind)",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"不应|不是|并非|并不|不能|不会|不再|不可能|不意味着|不自动|"
    r"不要|不可|绝非|不宜|没有|不存在|无需|无须|"
    r"避免|切勿|勿|错误(?:地|认为|表述)?|不适用"
)

_PAIRED_BOARD_RE = re.compile(
    r"(?:公共牌|公牌|牌面).{0,5}(?:成对|有一对|有对子|出现对子)|"
    r"paired board|成对牌面",
    re.IGNORECASE,
)
_PAIRED_BOARD_CAUSAL_TWO_PAIR_RE = re.compile(
    r"(?:(?:公共牌|公牌|牌面).{0,5}(?:成对|有一对|有对子|出现对子)|"
    r"paired board|成对牌面)"
    r".{0,12}(?:就|即|直接|意味着|形成|构成|算作|成为|已经是|已经成|视为|都是|都有)"
    r".{0,8}(?:两对|two pair)",
    re.IGNORECASE,
)
_TWO_PAIR_FROM_PAIRED_BOARD_RE = re.compile(
    r"(?:两对|two pair).{0,10}(?:来自|因为|源于).{0,8}"
    r"(?:(?:公共牌|公牌|牌面).{0,5}(?:成对|有一对|出现对子)|paired board)",
    re.IGNORECASE,
)
_HOLE_PAIR_CONDITION_RE = re.compile(
    r"口袋对|另一对|第二对|命中另一张|与.{0,4}配对|"
    r"(?:底牌|手牌|hole cards?).{0,10}(?:本身为?对子|形成一对|命中|配对|成对)|"
    r"同时.{0,6}(?:成对|对子)",
    re.IGNORECASE,
)
_DOUBLE_PAIRED_BOARD_RE = re.compile(
    r"两组对子|双对牌面|双重成对|double[- ]?paired|牌面本身.{0,4}两对",
    re.IGNORECASE,
)
_EXPLICIT_TWO_PAIR_STATE_RE = re.compile(
    r"(?:持有|已有|本身已有|成牌为|牌力为|确认为).{0,4}(?:两对|two pair)",
    re.IGNORECASE,
)
_PAIRED_BOARD_TWO_PAIR_NEGATION_RE = re.compile(
    r"(?:(?:公共牌|公牌|牌面).{0,5}(?:成对|有一对|有对子|出现对子)|"
    r"paired board|成对牌面)"
    r".{0,12}(?:不|并不|并非|不能|不会|不等于|不意味着|不自动)"
    r".{0,10}(?:两对|two pair)|"
    r"(?:两对|two pair).{0,12}(?:不|并不|并非|不能|不会|不等于)"
    r".{0,10}(?:(?:公共牌|公牌|牌面).{0,5}(?:成对|有一对|有对子|出现对子)|"
    r"paired board|成对牌面)",
    re.IGNORECASE,
)

_THREE_SUIT_BOARD_RE = re.compile(
    r"(?:(?:公共牌|公牌|牌面).{0,8}(?:三张|3张)(?:同一花色|相同花色|同花)|"
    r"三同花牌面|(?:三张|3张)同花(?:牌面)?)",
    re.IGNORECASE,
)
_FOUR_SUIT_BOARD_RE = re.compile(
    r"(?:(?:公共牌|公牌|牌面).{0,8}(?:四张|4张)(?:同一花色|相同花色|同花)|"
    r"四同花牌面|(?:四张|4张)同花(?:牌面)?)",
    re.IGNORECASE,
)
_MADE_HAND_CAUSAL = (
    r"(?:就|即|直接|意味着|形成|构成|已经是|已经成|成为|算作|都是|都有)"
)
_AUTO_THREE_SUIT_FLUSH_RE = re.compile(
    rf"{_THREE_SUIT_BOARD_RE.pattern}.{{0,12}}{_MADE_HAND_CAUSAL}.{{0,6}}"
    r"(?:同花(?!听牌|可能|威胁)|flush(?!\s+(?:draw|possible|threat)))",
    re.IGNORECASE,
)
_AUTO_FOUR_SUIT_FLUSH_RE = re.compile(
    rf"{_FOUR_SUIT_BOARD_RE.pattern}.{{0,12}}{_MADE_HAND_CAUSAL}.{{0,6}}"
    r"(?:同花(?!听牌|可能|威胁)|flush(?!\s+(?:draw|possible|threat)))",
    re.IGNORECASE,
)
_THREE_SUIT_HOLE_CONDITION_RE = re.compile(
    r"(?:底牌|手牌|hole cards?).{0,12}(?:两张|双张).{0,8}"
    r"(?:同一花色|相同花色|该花色|同花)|"
    r"(?:(?<!对手)(?<!对方)持有|(?:我方|我们|自己|本人|hero).{0,4}"
    r"(?:已有|成牌为|牌力为)).{0,5}(?:同花|flush)",
    re.IGNORECASE,
)
_FOUR_SUIT_HOLE_CONDITION_RE = re.compile(
    r"(?:底牌|手牌|hole cards?).{0,12}(?:至少)?(?:一张|任一张|有一张|含一张)"
    r".{0,6}(?:同一花色|相同花色|该花色)|"
    rf"{_THREE_SUIT_HOLE_CONDITION_RE.pattern}",
    re.IGNORECASE,
)
_FOUR_CONNECTED_BOARD_RE = re.compile(
    r"(?:(?:公共牌|公牌|牌面).{0,8}(?:四连|四张连续|四张相连|四张顺子)|"
    r"四连牌面)",
    re.IGNORECASE,
)
_AUTO_STRAIGHT_FROM_BOARD_RE = re.compile(
    rf"{_FOUR_CONNECTED_BOARD_RE.pattern}.{{0,12}}"
    rf"{_MADE_HAND_CAUSAL}.{{0,6}}"
    r"(?:顺子(?!听牌|可能|威胁)|straight(?!\s+(?:draw|possible|threat)))",
    re.IGNORECASE,
)
_HOLE_STRAIGHT_CONDITION_RE = re.compile(
    r"(?:底牌|手牌|hole cards?).{0,14}(?:补成|补足|完成|第五张)|"
    r"(?:(?<!对手)(?<!对方)持有|(?:我方|我们|自己|本人|hero).{0,4}"
    r"(?:已有|成牌为|牌力为)).{0,5}(?:顺子|straight)",
    re.IGNORECASE,
)
_PAIRED_BOARD_AUTO_TRIPS_RE = re.compile(
    rf"(?:{_PAIRED_BOARD_RE.pattern}).{{0,12}}"
    rf"{_MADE_HAND_CAUSAL}.{{0,6}}"
    r"(?:三条(?!听牌|可能|威胁)|trips|three of a kind)",
    re.IGNORECASE,
)
_PAIRED_BOARD_AUTO_FULL_HOUSE_RE = re.compile(
    rf"(?:{_PAIRED_BOARD_RE.pattern}).{{0,12}}"
    rf"{_MADE_HAND_CAUSAL}.{{0,6}}"
    r"(?:葫芦(?!听牌|可能|威胁)|full house)",
    re.IGNORECASE,
)
_HOLE_MATCHES_PAIRED_RANK_RE = re.compile(
    r"(?:底牌|手牌|hole cards?).{0,16}(?:与.{0,6}对子.{0,8}(?:同点|点数相同)|"
    r"命中.{0,6}对子|包含第三张|持有第三张|同点数牌)|"
    r"(?:(?<!对手)(?<!对方)持有|(?:我方|我们|自己|本人|hero).{0,4}"
    r"(?:已有|成牌为|牌力为)).{0,5}(?:三条|trips|three of a kind)",
    re.IGNORECASE,
)
_EXPLICIT_FULL_HOUSE_STATE_RE = re.compile(
    r"(?:(?<!对手)(?<!对方)持有|(?:我方|我们|自己|本人|hero).{0,4}"
    r"(?:已有|成牌为|牌力为)).{0,5}(?:葫芦|full house)",
    re.IGNORECASE,
)
_OPPONENT_SUBJECT_RE = re.compile(
    r"对手|对方|villain|opponent",
    re.IGNORECASE,
)
_HERO_SUBJECT_RE = re.compile(r"我方|我们|自己|本人|hero", re.IGNORECASE)
_RANGE_MADE_HAND_SUFFIX_RE = re.compile(
    r"^(?:的)?(?:组合|概率|可能性|风险|占比|范围)|"
    r"^\s+(?:combos?|combinations?|probability|possibility|risk|range)\b",
    re.IGNORECASE,
)
_HAND_CLASS_RANK = {
    "高牌": 0,
    "一对": 1,
    "两对": 2,
    "三条": 3,
    "顺子": 4,
    "同花": 5,
    "葫芦": 6,
    "四条": 7,
    "同花顺": 8,
    "皇家同花顺": 8,
}
_HAND_CLASS_TOKEN = "|".join(
    sorted(_HAND_CLASS_RANK, key=len, reverse=True)
)
_HAND_RANK_STRONGER_RE = re.compile(
    rf"(?P<left>{_HAND_CLASS_TOKEN})(?:的)?(?:牌型|牌力)?(?:上)?"
    r"(?:一定|总是|必然)?(?:强于|高于|胜过|击败)"
    rf"(?P<right>{_HAND_CLASS_TOKEN})"
)
_HAND_RANK_WEAKER_RE = re.compile(
    rf"(?P<left>{_HAND_CLASS_TOKEN})(?:的)?(?:牌型|牌力)?(?:上)?"
    r"(?:一定|总是|必然)?(?:弱于|低于|不如)"
    rf"(?P<right>{_HAND_CLASS_TOKEN})"
)

_POT_CONTROL_RE = re.compile(r"控池|控制底池|pot control", re.IGNORECASE)

_OPPONENT_RERAISE_RE = re.compile(
    r"(?:面对)?(?:对手|对方|villain|他)(?:的)?[^。；;]{0,10}"
    r"(?:再次加注|再加注|重新加注|re[- ]?raise|reraise|4[- ]?bet)",
    re.IGNORECASE,
)
_HERO_RAISE_RE = re.compile(
    r"(?:我方|我们|自己|本人|hero)[^。；;]{0,12}"
    r"(?:加注|3[- ]?bet|三bet|回加|反加)|(?:我方|我们的?)(?:加注|3[- ]?bet)|"
    r"回加|反加",
    re.IGNORECASE,
)
_THIRD_PARTY_RAISE_RE = re.compile(
    r"(?:另一名|第三名|第三位|后位|其他)玩家[^。；;]{0,10}"
    r"(?:加注|3[- ]?bet)",
    re.IGNORECASE,
)
_STREET_MARKER_RE = re.compile(
    r"翻牌前|翻牌|转牌|河牌|\bpre[- ]?flop\b|\bflop\b|\bturn\b|\briver\b",
    re.IGNORECASE,
)
_PREFLOP_CONTEXT_RE = re.compile(
    r"翻牌前|起手牌|看翻牌|limp|跛入|开池|open[- ]?raise|"
    r"(?:2|3|4|5)[- ]?bet|\bpre[- ]?flop\b",
    re.IGNORECASE,
)
_PROJECTED_FLOP_SPR_RE = re.compile(
    r"(?:预计|预期|目标|计划|控制|使|让).{0,8}(?:翻牌|flop).{0,6}SPR|"
    r"(?:翻牌|flop)(?:后|时|圈)?(?:的)?\s*SPR",
    re.IGNORECASE,
)
_STACK_DEPTH_SPR_EQUIVALENCE_RE = re.compile(
    r"深筹码\s*[（(]\s*SPR\s*(?:高|大)\s*[）)]|"
    r"(?:深筹码|有效筹码深).{0,6}(?:等同于|就是|即|意味着).{0,4}高?\s*SPR|"
    r"高?\s*SPR.{0,6}(?:等同于|就是|即|意味着).{0,4}(?:深筹码|有效筹码深)",
    re.IGNORECASE,
)
_AMBIGUOUS_PAIRED_HIGH_CARD_RE = re.compile(
    r"(?:成对|对子)[2-9TJQKA](?:高牌|high)|[2-9TJQKA](?:高牌|high)(?:成对|对子)",
    re.IGNORECASE,
)
_STRAIGHT_DRAW_RE = re.compile(r"顺子听牌|straight\s+draw", re.IGNORECASE)
_GUTSHOT_RE = re.compile(r"卡顺|gutshot|inside\s+straight", re.IGNORECASE)
_OPEN_ENDED_RE = re.compile(
    r"两头顺|双头顺|开放式顺子听牌|open[- ]?ended", re.IGNORECASE
)
_DOUBLE_GUTSHOT_RE = re.compile(r"双卡顺|double\s+gutshot", re.IGNORECASE)
_FOUR_OUTS_RE = re.compile(
    r"(?<!\d)4\s*(?:张|个)?\s*(?:理论|名义|干净|clean)?\s*outs?",
    re.IGNORECASE,
)
_EIGHT_OUTS_RE = re.compile(
    r"(?<!\d)8\s*(?:张|个)?\s*(?:理论|名义|干净|clean)?\s*outs?",
    re.IGNORECASE,
)
_OUTS_QUALIFIER_RE = re.compile(r"理论|名义|干净|clean|脏\s*outs?", re.IGNORECASE)
_FLOP_CONTEXT_RE = re.compile(
    r"翻后|翻牌(?!前)|\bflop\b",
    re.IGNORECASE,
)
_TURN_CONTEXT_RE = re.compile(r"转牌圈?|\bturn\b", re.IGNORECASE)
_RIVER_CONTEXT_RE = re.compile(r"河牌圈?|\briver\b", re.IGNORECASE)
_FREE_RIVER_RE = re.compile(r"免费(?:看|见|到)?河牌|免费看河牌", re.IGNORECASE)
_FREE_SHOWDOWN_RE = re.compile(r"免费摊牌|免费看摊牌|直接进入摊牌", re.IGNORECASE)
_FREE_FLOP_RE = re.compile(r"免费(?:看|见|到)?翻牌|免费看翻牌", re.IGNORECASE)
_FREE_TURN_RE = re.compile(r"免费(?:看|见|到)?转牌|免费看转牌", re.IGNORECASE)
_HERO_CHECK_BEFORE_FREE_CARD_RE = re.compile(
    r"双方过牌|(?:我方|我们|自己|本人|hero).{0,6}过牌|"
    r"(?:BB|大盲位?|big blind).{0,6}过牌|"
    r"(?:可|应|选择|继续|随后|然后)\s*过牌",
    re.IGNORECASE,
)
_HU_BUTTON_OWN_BLIND_ACTION_RE = re.compile(
    r"(?:面对|跟注|平跟|call(?:ing)?).{0,5}(?:小盲(?:注)?|small blind\b)",
    re.IGNORECASE,
)
_HU_BUTTON_CALL_LIMP_RE = re.compile(
    r"(?:BTN\s*/\s*SB|SB\s*/\s*BTN)[^。；;\n]{0,100}"
    r"(?:(?:跟注|平跟|call(?:ing)?)[^。；;\n]{0,6}(?:limp|跛入)|"
    r"面对[^。；;\n]{0,8}(?:BB|大盲位?|big blind|对手|对方|opponent)"
    r"[^。；;\n]{0,8}"
    r"(?:的\s*)?(?:limp|跛入))",
    re.IGNORECASE,
)
_FACING_UNRESOLVED_BET_RE = re.compile(
    r"(?:面对|遭遇)[^，,。；;\n]{0,8}?(?:下注|bet)|"
    r"(?:若|如果|当)(?:对手|对方|villain)(?:的)?"
    r"(?:(?:小额|大额|较小|较大|持续|再次|半池|全池|满池))?(?:下注|bet)|"
    r"(?:对手|对方|villain)(?:的)?"
    r"(?:(?:小额|大额|较小|较大|持续|再次|半池|全池|满池))?"
    r"(?:下注|bet)(?:时|后)",
    re.IGNORECASE,
)
_ILLEGAL_FACING_BET_RESPONSE_RE = re.compile(
    r"(?:应|应该|可|可以|选择|优先|考虑|则|改为|转为)"
    r"[^。；;\n]{0,8}(?P<action>下注|过牌)|"
    r"(?:加注|跟注|弃牌)\s*(?:或|/)\s*(?P<menu_action>下注|过牌)",
    re.IGNORECASE,
)
_OPPONENT_CHECK_BRANCH_RE = re.compile(
    r"(?:若|如果|当)?(?:对手|对方|villain)[^。；;\n]{0,4}过牌",
    re.IGNORECASE,
)
_HU_SINGLE_OPPONENT_PURPOSE_RE = re.compile(
    r"隔离(?:对手|玩家)?|争取单挑|形成单挑|打成单挑|isolate|isolation",
    re.IGNORECASE,
)
_BOARD_DRAW_THREAT_RE = re.compile(
    r"(?:公共牌|公牌|牌面).{0,10}(?:同花|顺子).{0,4}(?:听牌|可能|威胁|结构)",
    re.IGNORECASE,
)
_ANONYMOUS_DRAW_RESULT_RE = re.compile(
    r"(?:未|没有|若未|如果未).{0,6}(?:完成|击中|补成|改善).{0,4}(?:听牌|同花|顺子)|"
    r"(?:听牌|同花|顺子).{0,4}(?:未完成|未击中|未补成)",
    re.IGNORECASE,
)
_EXPLICIT_HERO_DRAW_RE = re.compile(
    r"(?:我方|我们|自己|本人|hero|底牌|手牌).{0,12}(?:同花|顺子)?听牌|"
    r"(?:我方|我们|自己|本人|hero).{0,12}(?:听同花|听顺子)|"
    r"持有.{0,10}(?:同花|顺子)听牌",
    re.IGNORECASE,
)
_AMBIGUOUS_PAIRED_STATE_RE = re.compile(
    r"(?:(?:翻牌|转牌|河牌)(?:圈|后)?\s*)?成对(?=且|但|时|，|,|$)|"
    r"持有(?:一对|对子)",
    re.IGNORECASE,
)
_HERO_OR_BOARD_PAIR_SUBJECT_RE = re.compile(
    r"(?:我方|我们|自己|本人|hero|底牌|手牌|公共牌|公牌|牌面).{0,12}"
    r"(?:成对|持有一对|持有对子|一对|对子)",
    re.IGNORECASE,
)
_OPPONENT_PRIOR_CHECK_RE = re.compile(
    r"(?:对手|对方|BB|大盲位?|big blind|villain|opponent).{0,8}"
    r"(?:先|已经|已)?过牌|双方过牌",
    re.IGNORECASE,
)
_PREFLOP_SPECIFIC_ACTION_RE = re.compile(
    r"跟注|平跟|弃牌|加注|过牌|免费看翻牌|看翻牌|limp|call|fold|raise|check",
    re.IGNORECASE,
)
_SPECIFIC_STARTING_HAND_RE = re.compile(
    r"(?:垃圾牌|极弱牌|弱牌|中等牌力|中等强度牌|强牌|强起手牌|口袋对子|"
    r"同花高张|非同花高张|连接牌|间隔牌|起手牌范围|入池范围)",
    re.IGNORECASE,
)
_INFORMATION_BET_RE = re.compile(
    r"(?:下注|加注).{0,20}(?:获取|获得|收集|试探).{0,4}信息|"
    r"(?:获取|获得|收集|试探).{0,4}信息.{0,20}(?:下注|加注)",
    re.IGNORECASE,
)
_BET_POT_CONTROL_RE = re.compile(
    r"(?:下注|加注).{0,20}(?:控池|控制底池|pot control)|"
    r"(?:控池|控制底池|pot control).{0,20}(?:下注|加注)",
    re.IGNORECASE,
)
_NONCOMMITTAL_ACTION_MENU_RE = re.compile(
    r"(?:可|可以|应考虑).{0,12}(?:下注|加注|跟注|过牌|弃牌).{0,8}"
    r"(?:或|/).{0,8}(?:下注|加注|跟注|过牌|弃牌).{0,16}"
    r"(?:根据|视|取决于).{0,8}(?:牌力|情况|局面|对手)",
    re.IGNORECASE,
)
_DIRECT_ACTION_MENU_RE = re.compile(
    r"(?P<first>下注|加注|跟注|过牌|弃牌)[^。；;\n]{0,4}"
    r"(?:或|/)[^。；;\n]{0,4}"
    r"(?P<second>下注|加注|跟注|过牌|弃牌)",
    re.IGNORECASE,
)
_LEGAL_ACTION_ENUMERATION_RE = re.compile(
    r"只能[^。；;\n]{0,24}(?:弃牌|跟注|加注)[^。；;\n]{0,16}不能",
    re.IGNORECASE,
)
_BARE_ACTION_MENU_RE = re.compile(
    r"(?:可|可以|应|优先)\s*(?:继续|选择|考虑)?\s*"
    r"(?:下注|加注|跟注|过牌|弃牌)\s*(?:或|/)\s*"
    r"(?:谨慎|选择|继续|小额|大额)?\s*"
    r"(?:下注|加注|跟注|过牌|弃牌)",
    re.IGNORECASE,
)
_EXPLICIT_MENU_BRANCH_RE = re.compile(
    r"若|如果|否则|反之|分别|仅当|除非|"
    r"(?:底池赔率|所需权益|下注尺度|跟注价格).{0,12}(?:阈值|高于|低于|不少于|不高于|\d)",
    re.IGNORECASE,
)
_VAGUE_CAUTION_RE = re.compile(
    r"(?:应|需|要)?(?:保持)?谨慎.{0,12}(?:避免)?过度投入|"
    r"避免过度投入(?:筹码)?(?:[。；;]|$)|"
    r"(?:必要时|视情况|酌情)(?:选择|考虑)?(?:下注|加注|跟注|过牌|弃牌)"
)
_RIVER_FUTURE_DRAW_RE = re.compile(
    r"(?:河牌圈?|\briver\b)[^。；;\n]{0,80}"
    r"(?:更高|更强|其他|新的)[^。；;\n]{0,8}(?:听牌|draw)[^。；;\n]{0,8}(?:可能|潜力|机会)?|"
    r"(?:河牌圈?|\briver\b)[^。；;\n]{0,80}牌面[^。；;\n]{0,12}"
    r"(?:听牌|draw)[^。；;\n]{0,8}(?:可能|潜力|机会)",
    re.IGNORECASE,
)
_HERO_RAISE_RECOMMENDATION_RE = re.compile(
    r"(?:应|应该|可|可以|优先|选择|考虑)[^。；;\n]{0,16}(?:加注|raise)",
    re.IGNORECASE,
)
_OPPONENT_BET_OR_RAISE_RE = re.compile(
    r"(?:对手|对方|villain|opponent)[^。；;\n]{0,16}(?:下注|加注|bet|raise)|"
    r"(?:面对|遭遇)[^。；;\n]{0,16}(?:下注|加注|bet|raise)|"
    r"(?:过牌\s*[-—–]?\s*加注|check\s*[- ]?raise)",
    re.IGNORECASE,
)
_HERO_RECOMMENDED_ACTION_RE = re.compile(
    r"(?:应|应该|可|可以|优先|选择|则|改为|转为|采取|继续)"
    r"[^。；;\n]{0,18}(?P<action>弃牌|过牌|跟注|加注|下注)",
    re.IGNORECASE,
)
_HU_BB_OPPONENT_CHECK_RAISE_RE = re.compile(
    r"(?:对手|对方|BTN\s*/\s*SB|SB\s*/\s*BTN)"
    r"[^。；;\n]{0,24}(?:翻牌(?:后|圈)?|flop)?[^。；;\n]{0,12}"
    r"(?:过牌\s*[-—–]?\s*加注|check\s*[- ]?raise)",
    re.IGNORECASE,
)
_STATIC_ACTION_LEGALITY_RE = re.compile(
    r"面对(?:对手|对方)?[^。；;\n]{0,8}(?:下注|bet)时?"
    r"[^。；;\n]{0,8}只能[^。；;\n]{0,24}(?:弃牌|跟注|加注)"
    r"[^。；;\n]{0,16}不能[^。；;\n]{0,8}(?:下注|bet)",
    re.IGNORECASE,
)
_DETERMINISTIC_OPPONENT_HAND_INFERENCE_RE = re.compile(
    r"(?:对手|对方|其)[^。；;\n]{0,18}(?:加注|下注|过牌)"
    r"[^。；;\n]{0,10}(?:说明|表明|意味着|证明)"
    r"[^。；;\n]{0,8}(?:其|对手|对方)?(?:持有|有)强牌",
    re.IGNORECASE,
)
_AMBIGUOUS_POKER_TERM_RE = re.compile(
    r"(?P<high_pair>(?<!超)(?<!顶)高对)|(?P<pay_nuts>支付坚果)",
    re.IGNORECASE,
)
_UNDEFINED_STRATEGY_THRESHOLD_RE = re.compile(
    r"(?:在)?(?:合适时机|有利情况下|情况合适时)"
    r"[^。；;\n]{0,12}(?:弃牌|过牌|跟注|加注|下注)|"
    r"(?:谨慎|保守地?)\s*(?:弃牌|过牌|跟注|加注|下注)",
    re.IGNORECASE,
)
_OBSERVABLE_STRATEGY_THRESHOLD_RE = re.compile(
    r"底池赔率|所需权益|跟注价格|下注尺度|底池比例|有效筹码|SPR|"
    r"牌面|公共牌|位置|阻断牌|组合|范围|频率|倾向|听牌|成牌|顶对|超对|"
    r"两对|三条|顺子|同花|葫芦|四条|同花顺|诈唬|价值|"
    r"\d+(?:\.\d+)?\s*(?:BB|bb|%|倍|池)",
    re.IGNORECASE,
)
_REVERSAL_MARKER_RE = re.compile(r"(?:反转|例外)\s*[:：]", re.IGNORECASE)
_HERO_SIZING_CHANGE_RE = re.compile(
    r"(?:应|应该|则|改为|转为|选择|优先|增加|减少|提高|降低|扩大|缩小)"
    r"[^。；;\n]{0,18}(?:下注|加注|跟注|弃牌|过牌|尺度|频率)",
    re.IGNORECASE,
)
_CHECK_FOR_SHOWDOWN_RE = re.compile(
    r"(?:应|应该|可|可以|优先|选择|继续)[^。；;\n]{0,12}过牌"
    r"[^。；;\n]{0,12}(?:争取|获得|实现)[^。；;\n]{0,8}(?:便宜|低成本)?摊牌|"
    r"过牌[^。；;\n]{0,12}(?:争取|获得|实现)[^。；;\n]{0,8}"
    r"(?:便宜|低成本)?摊牌[^。；;\n]{0,10}(?:合理|优先|首选)",
    re.IGNORECASE,
)
_OPPONENT_ALREADY_CHECKED_RE = re.compile(
    r"(?:面对|在)?[^。；;\n]{0,8}(?:对手|对方|BB|大盲位?|villain|opponent)"
    r"[^。；;\n]{0,8}(?:先|已经|已)?过牌|"
    r"(?:对手|对方|BB|大盲位?|villain|opponent)"
    r"[^。；;\n]{0,6}(?:check(?:ed)?|先过牌)",
    re.IGNORECASE,
)
_CONFLICTING_CHECK_ARCHETYPES_RE = re.compile(
    r"check\s*[-—–]?\s*call[^。；;\n]{0,12}(?:或|或者|和|与|/)"
    r"[^。；;\n]{0,12}check\s*[-—–]?\s*fold|"
    r"check\s*[-—–]?\s*fold[^。；;\n]{0,12}(?:或|或者|和|与|/)"
    r"[^。；;\n]{0,12}check\s*[-—–]?\s*call",
    re.IGNORECASE,
)
_CHECK_FOLD_RE = re.compile(r"check\s*[-—–]?\s*fold", re.IGNORECASE)
_VALUE_BET_PURPOSE_RE = re.compile(
    r"价值下注|下注[^。；;\n]{0,10}(?:榨取|获取|获得)价值|"
    r"(?:榨取|获取|获得)[^。；;\n]{0,8}价值",
    re.IGNORECASE,
)
_FACING_AGGRESSION_RE = re.compile(
    r"(?:面对|遭遇)[^。；;\n]{0,16}(?:对手|对方|villain|opponent)?"
    r"[^。；;\n]{0,10}(?:下注|加注|bet|raise)",
    re.IGNORECASE,
)
_FOLD_OR_CALL_RECOMMENDATION_RE = re.compile(
    r"(?:应|应该|则|优先|选择|倾向于?|可以|可)[^。；;\n]{0,14}"
    r"(?:弃牌|跟注|fold|call)",
    re.IGNORECASE,
)
_HERO_MARGINAL_HAND_RE = re.compile(
    r"(?:我方|自己|本人|hero)[^。；;\n]{0,8}(?:仅|只)?(?:持有|持)?"
    r"[^。；;\n]{0,8}(?:底对|中对|弱对|边缘牌力|弱听牌|无听牌)|"
    r"(?:若|当)?(?:仅|只)持(?:有)?[^。；;\n]{0,8}"
    r"(?:底对|中对|弱对|边缘牌力|弱听牌|无听牌)",
    re.IGNORECASE,
)
_AGGRESSION_PRICE_OR_EQUITY_RE = re.compile(
    r"底池赔率|pot odds|跟注价格|跟注成本|所需权益|估计权益|"
    r"权益[^。；;\n]{0,8}(?:高于|低于|不足|超过)|下注尺度|加注尺度|"
    r"有效筹码|\bSPR\b|(?:半池|满池|超池|四分之[一三]|三分之[一二]|"
    r"\d+(?:\.\d+)?\s*(?:%|倍|BB|bb|池))|"
    r"(?:下注|加注)到[^。；;\n]{0,12}(?:底池|BB|bb|筹码)",
    re.IGNORECASE,
)
_POSTFLOP_CONTEXT_RE = re.compile(
    r"翻牌后|翻后|翻牌圈|转牌圈?|河牌圈?|\bflop\b|\bturn\b|\briver\b",
    re.IGNORECASE,
)
_HERO_CHECK_RECOMMENDATION_RE = re.compile(
    r"(?:应|应该|可|可以|优先|选择|继续)[^。；;\n]{0,14}过牌",
    re.IGNORECASE,
)
_CHECK_AVAILABLE_STATE_RE = re.compile(
    r"(?:我方|自己|本人|hero)[^。；;\n]{0,8}(?:先行动|率先行动|首先行动)|"
    r"(?:在|处于)?(?:BB|大盲位?|无位置|OOP)[^。；;\n]{0,10}"
    r"(?:先行动|率先行动|首先行动)?|"
    r"(?:对手|对方|BB|大盲位?|villain|opponent)[^。；;\n]{0,8}"
    r"(?:先|已经|已)?过牌",
    re.IGNORECASE,
)
_STATIC_HU_ORDER_RULE_RE = re.compile(
    r"(?:HU|两人桌|双人桌|单挑)[^。；;\n]{0,24}"
    r"(?:BTN\s*/\s*SB|SB\s*/\s*BTN)[^。；;\n]{0,24}"
    r"翻牌前[^。；;\n]{0,12}先行动[^。；;\n]{0,40}"
    r"翻牌后[^。；;\n]{0,12}后(?:于|行动)[^。；;\n]{0,40}"
    r"(?:BB|大盲位?)[^。；;\n]{0,16}翻牌后[^。；;\n]{0,12}先行动",
    re.IGNORECASE,
)
_CHECK_CALL_LINE_RE = re.compile(
    r"check\s*[-—–]?\s*call|过牌\s*[-—–]?\s*跟注",
    re.IGNORECASE,
)
_CHECK_CALL_STATE_RE = re.compile(
    r"(?:BB|大盲位?|无位置|OOP|先行动|率先行动|首先行动)|"
    r"(?:我方|自己|本人|hero)[^。；;\n]{0,8}过牌"
    r"[^。；;\n]{0,18}(?:对手|对方|villain|opponent)[^。；;\n]{0,8}下注",
    re.IGNORECASE,
)


def validate_poker_experience_document(body: str) -> list[PokerExperienceIssue]:
    """Return deterministic high-confidence poker-semantic issues in Markdown bullets."""

    issues: list[PokerExperienceIssue] = []
    for section, bullet in _iter_markdown_bullets(body):
        issues.extend(_validate_bullet(section, bullet))
    return issues


def _iter_markdown_bullets(body: str) -> Iterator[tuple[str, str]]:
    section = ""
    for raw_line in body.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        heading = _HEADING_RE.match(raw_line)
        if heading:
            section = heading.group(1).strip()
            continue
        bullet = _BULLET_RE.match(raw_line)
        if bullet:
            yield section, bullet.group(1).strip()


def _validate_bullet(section: str, bullet: str) -> list[PokerExperienceIssue]:
    issues: list[PokerExperienceIssue] = []
    issues.extend(_validate_suit_suffixes(section, bullet))
    issues.extend(_validate_hyphenated_hole_cards(section, bullet))
    issues.extend(_validate_hand_notation_order(section, bullet))
    issues.extend(_validate_connectivity_labels(section, bullet))
    issues.extend(_validate_position_street_scope(section, bullet))
    issues.extend(_validate_hu_preflop_position(section, bullet))
    issues.extend(_validate_hu_button_blind_action(section, bullet))
    issues.extend(_validate_hu_button_limp_call(section, bullet))
    issues.extend(_validate_hu_postflop_position(section, bullet))
    issues.extend(_validate_hu_ip_check_order(section, bullet))
    issues.extend(_validate_hu_bb_opponent_check_raise(section, bullet))
    issues.extend(_validate_spr_usage(section, bullet))
    issues.extend(_validate_draw_outs(section, bullet))
    issues.extend(_validate_free_card_street(section, bullet))
    issues.extend(_validate_free_card_action_chain(section, bullet))
    issues.extend(_validate_facing_bet_response(section, bullet))
    issues.extend(_validate_betting_purpose(section, bullet))
    issues.extend(_validate_noncommittal_action(section, bullet))
    issues.extend(_validate_static_action_legality(section, bullet))
    issues.extend(_validate_deterministic_opponent_hand_inference(section, bullet))
    issues.extend(_validate_undefined_strategy_threshold(section, bullet))
    issues.extend(_validate_reversal_branch(section, bullet))
    issues.extend(_validate_check_showdown_state(section, bullet))
    issues.extend(_validate_conflicting_opponent_archetypes(section, bullet))
    issues.extend(_validate_check_fold_value_purpose(section, bullet))
    issues.extend(_validate_facing_aggression_price(section, bullet))
    issues.extend(_validate_postflop_check_state(section, bullet))
    issues.extend(_validate_check_call_state(section, bullet))
    issues.extend(_validate_static_invariant_experience(section, bullet))
    issues.extend(_validate_ambiguous_poker_terms(section, bullet))
    issues.extend(_validate_hu_isolation(section, bullet))
    issues.extend(_validate_ambiguous_hand_terms(section, bullet))
    issues.extend(_validate_ambiguous_paired_state(section, bullet))
    issues.extend(_validate_board_threat_draw_subject(section, bullet))
    issues.extend(_validate_paired_board_two_pair(section, bullet))
    issues.extend(_validate_board_made_hand_claims(section, bullet))
    issues.extend(_validate_hand_ranking(section, bullet))
    issues.extend(_validate_river_pot_control(section, bullet))
    issues.extend(_validate_river_future_draw(section, bullet))
    issues.extend(_validate_postflop_raise_requires_facing_action(section, bullet))
    issues.extend(_validate_reraise_chain(section, bullet))
    issues.extend(_validate_preflop_section_placement(section, bullet))
    issues.extend(_validate_later_street_section_placement(section, bullet))
    return _deduplicate_issues(issues)


def _validate_position_street_scope(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Require heads-up BTN/SB position claims to state the betting street.

    BTN/SB acts first preflop but last postflop.  A bare label such as
    ``BTN/SB（有利位置）`` silently collapses those two opposite action orders.
    """

    for clause in re.split(r"[。；;\n]", bullet):
        if (
            _HU_COMBINED_BUTTON_RE.search(clause)
            and _POSITION_POLARITY_RE.search(clause)
            and not _EXPLICIT_PREFLOP_RE.search(clause)
            and not _EXPLICIT_POSTFLOP_RE.search(clause)
        ):
            return [
                PokerExperienceIssue(
                    code="hu_position_street_ambiguous",
                    section=section,
                    bullet=bullet,
                    message=(
                        "两人桌 BTN/SB 翻牌前先行动、翻牌后后行动；位置结论必须"
                        "明确街道，不能笼统标为有利或不利位置。"
                    ),
                )
            ]
    return []


def _validate_hu_preflop_position(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Validate action order and position polarity inside each preflop clause."""

    issues: list[PokerExperienceIssue] = []
    for clause in re.split(r"[。；;\n]", bullet):
        preflop = _EXPLICIT_PREFLOP_RE.search(clause)
        if preflop is None:
            continue
        postflop = _EXPLICIT_POSTFLOP_RE.search(clause, preflop.end())
        scoped = clause[: postflop.start() if postflop else len(clause)]
        combined_roles = list(_HU_COMBINED_BUTTON_RE.finditer(scoped))
        btn_roles = [
            role
            for role in _BTN_RE.finditer(scoped)
            if not any(
                combined.start() <= role.start() and role.end() <= combined.end()
                for combined in combined_roles
            )
        ]
        sb_roles = [
            role
            for role in _SB_RE.finditer(scoped)
            if not any(
                combined.start() <= role.start() and role.end() <= combined.end()
                for combined in combined_roles
            )
        ]
        bb_roles = list(_BB_RE.finditer(scoped))
        all_roles = sorted(
            [*combined_roles, *btn_roles, *sb_roles, *bb_roles],
            key=lambda item: item.start(),
        )
        button_roles = list(combined_roles)
        if _TWO_PLAYER_TABLE_RE.search(bullet):
            button_roles.extend([*btn_roles, *sb_roles])

        button_conflict = any(
            (
                claim := _nearest_direct_claim(
                    role, _BTN_PREFLOP_WRONG_RE, all_roles, scoped
                )
            )
            is not None
            and claim.start() >= preflop.end()
            and not _position_claim_is_negated(scoped, role, claim)
            for role in button_roles
        )
        if button_conflict:
            issues.append(
                PokerExperienceIssue(
                    code="hu_btn_preflop_position_conflict",
                    section=section,
                    bullet=bullet,
                    message=(
                        "两人桌 BTN/SB 翻牌前先行动，不能称为翻前有位置优势或"
                        "最后行动；其位置优势属于翻牌后。"
                    ),
                )
            )
        bb_conflict = _TWO_PLAYER_TABLE_RE.search(bullet) and any(
            (
                claim := _nearest_direct_claim(
                    role, _BB_PREFLOP_WRONG_RE, all_roles, scoped
                )
            )
            is not None
            and claim.start() >= preflop.end()
            and not _position_claim_is_negated(scoped, role, claim)
            for role in bb_roles
        )
        if bb_conflict:
            issues.append(
                PokerExperienceIssue(
                    code="hu_bb_preflop_position_conflict",
                    section=section,
                    bullet=bullet,
                    message="两人桌 BB 翻牌前最后行动，不能称为翻前无位置或先行动。",
                )
            )
    return issues


def _validate_hand_notation_order(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Require canonical high-rank-first starting-hand notation."""

    issues: list[PokerExperienceIssue] = []
    for hand in _RANKED_HAND_RE.finditer(bullet):
        first = hand.group("high").upper()
        second = hand.group("low").upper()
        if first == second or _RANK_VALUE[first] > _RANK_VALUE[second]:
            continue
        issues.append(
            PokerExperienceIssue(
                code="noncanonical_hand_rank_order",
                section=section,
                bullet=bullet,
                message=(
                    f"{hand.group(0)} 应按高点数在前书写为 "
                    f"{second}{first}{hand.group('suffix')}。"
                ),
            )
        )
    return issues


def _validate_hyphenated_hole_cards(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Reject rank-pair examples written as dealt-card literals such as ``9-3o``."""

    issues: list[PokerExperienceIssue] = []
    for hand in _HYPHENATED_HOLE_CARD_RE.finditer(bullet):
        if _NOTATION_META_NEGATION_RE.search(bullet[max(0, hand.start() - 24) : hand.start()]):
            continue
        first = hand.group("first").upper()
        second = hand.group("second").upper()
        ordered = sorted((first, second), key=_RANK_VALUE.__getitem__, reverse=True)
        raw_suffix = hand.group("suffix").lower()
        suffix = "s" if raw_suffix in {"s", "同花"} else "o"
        issues.append(
            PokerExperienceIssue(
                code="hyphenated_hole_card_notation",
                section=section,
                bullet=bullet,
                message=(
                    f"{hand.group(0)} 应使用规范起手牌记法 "
                    f"{ordered[0]}{ordered[1]}{suffix}，不应写成带连字符的两张牌。"
                ),
            )
        )
    return issues


def _validate_hu_button_blind_action(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """The heads-up button posts the small blind; it cannot face or call that blind."""

    for clause in re.split(r"[。；;\n]", bullet):
        if not _HU_COMBINED_BUTTON_RE.search(clause):
            continue
        conflict = _HU_BUTTON_OWN_BLIND_ACTION_RE.search(clause)
        if conflict is None or _match_is_negated(clause, conflict):
            continue
        return [
            PokerExperienceIssue(
                code="hu_button_own_blind_action_conflict",
                section=section,
                bullet=bullet,
                message=(
                    "两人桌 BTN 同时是 SB 并由自己投入小盲；不能把小盲注写成"
                    "其面对或跟注的对手动作。"
                ),
            )
        ]
    return []


def _validate_hu_button_limp_call(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """HU BTN/SB may complete its own blind, but cannot face another player's limp."""

    conflict = _HU_BUTTON_CALL_LIMP_RE.search(bullet)
    if conflict is None or _match_is_negated(bullet, conflict):
        return []
    return [
        PokerExperienceIssue(
            code="hu_button_limp_call_conflict",
            section=section,
            bullet=bullet,
            message=(
                "两人桌 BTN/SB 翻牌前首先行动；未加注时只能弃牌、补齐小盲或"
                "加注，不能写成面对并跟注 BB 的 limp。"
            ),
        )
    ]


def _validate_facing_bet_response(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """A live bet must be answered by fold, call, or raise—not bet/check."""

    for clause in re.split(r"[。；;\n]", bullet):
        facing = _FACING_UNRESOLVED_BET_RE.search(clause)
        if facing is None or _match_is_negated(clause, facing):
            continue
        for response in _ILLEGAL_FACING_BET_RESPONSE_RE.finditer(
            clause, facing.end()
        ):
            if _match_is_negated(clause, response):
                continue
            between = clause[facing.end() : response.start()]
            if _OPPONENT_CHECK_BRANCH_RE.search(between):
                continue
            if _STREET_MARKER_RE.search(between):
                continue
            action = response.group("action") or response.group("menu_action")
            return [
                PokerExperienceIssue(
                    code="facing_bet_illegal_response",
                    section=section,
                    bullet=bullet,
                    message=(
                        f"面对尚未结清的对手下注时不能选择{action}；合法响应只能是"
                        "弃牌、跟注或加注。若描述另一分支，必须明确写出对手先过牌。"
                    ),
                )
            ]
    return []


def _validate_spr_usage(section: str, bullet: str) -> list[PokerExperienceIssue]:
    """Reject current-SPR language in an explicitly preflop decision."""

    if "spr" not in bullet.lower():
        return []
    issues: list[PokerExperienceIssue] = []
    equivalence = _STACK_DEPTH_SPR_EQUIVALENCE_RE.search(bullet)
    if equivalence is not None and not _match_is_negated(bullet, equivalence):
        issues.append(
            PokerExperienceIssue(
                code="stack_depth_spr_equivalence",
                section=section,
                bullet=bullet,
                message="深筹码是绝对有效筹码深度，不等同于翻牌后的高 SPR。",
            )
        )
    preflop_context = section.strip() == "起手牌" or _PREFLOP_CONTEXT_RE.search(bullet)
    if preflop_context and not _PROJECTED_FLOP_SPR_RE.search(bullet):
        issues.append(
            PokerExperienceIssue(
                code="preflop_spr_misuse",
                section=section,
                bullet=bullet,
                message=(
                    "SPR 是翻牌后有效筹码与底池之比；翻牌前只能描述预计/目标翻牌 "
                    "SPR，不能把当前有效筹码深度直接写成 SPR。"
                ),
            )
        )
    elif (
        equivalence is None
        and not _EXPLICIT_POSTFLOP_RE.search(bullet)
        and not _PROJECTED_FLOP_SPR_RE.search(bullet)
    ):
        issues.append(
            PokerExperienceIssue(
                code="spr_street_undefined",
                section=section,
                bullet=bullet,
                message="当前 SPR 只在翻牌后定义；规则必须明确翻牌后街道。",
            )
        )
    return issues


def _validate_draw_outs(section: str, bullet: str) -> list[PokerExperienceIssue]:
    """Require a named straight-draw type before assigning a numeric out count."""

    if not any(
        pattern.search(bullet)
        for pattern in (
            _STRAIGHT_DRAW_RE,
            _GUTSHOT_RE,
            _OPEN_ENDED_RE,
            _DOUBLE_GUTSHOT_RE,
        )
    ):
        return []
    issues: list[PokerExperienceIssue] = []
    for value, outs_re in ((4, _FOUR_OUTS_RE), (8, _EIGHT_OUTS_RE)):
        for outs in outs_re.finditer(bullet):
            if _match_is_negated(bullet, outs):
                continue
            draw_kind = _nearest_straight_draw_kind(bullet, outs)
            if draw_kind is None or draw_kind == "generic":
                issues.append(
                    PokerExperienceIssue(
                        code="straight_draw_outs_underspecified",
                        section=section,
                        bullet=bullet,
                        message=(
                            f"不能把泛称顺子听牌固定为 {value} outs；应明确卡顺、"
                            "两头顺或双卡顺等听牌类型。"
                        ),
                    )
                )
            elif (draw_kind == "gutshot" and value != 4) or (
                draw_kind in {"open_ended", "double_gutshot"} and value != 8
            ):
                expected = 4 if draw_kind == "gutshot" else 8
                issues.append(
                    PokerExperienceIssue(
                        code="straight_draw_outs_conflict",
                        section=section,
                        bullet=bullet,
                        message=(
                            f"该听牌通常有 {expected} 张理论 outs，不是 {value} 张。"
                        ),
                    )
                )
            clause_start, clause_end = _clause_bounds(
                bullet, outs.start(), "。；;\n"
            )
            qualifier_context = bullet[
                max(clause_start, outs.start() - 12) : min(
                    clause_end, outs.end() + 12
                )
            ]
            if not _OUTS_QUALIFIER_RE.search(qualifier_context):
                issues.append(
                    PokerExperienceIssue(
                        code="draw_outs_missing_quality_qualifier",
                        section=section,
                        bullet=bullet,
                        message="数字 outs 应标为理论/名义或干净 outs，并考虑脏 outs。",
                    )
                )
    return issues


def _nearest_straight_draw_kind(
    text: str, outs: re.Match[str]
) -> str | None:
    """Bind one numeric outs claim to the nearest named draw in its sentence."""

    clause_start, clause_end = _clause_bounds(text, outs.start(), "。；;\n")
    candidates: list[tuple[int, int, str]] = []
    specs = (
        ("double_gutshot", _DOUBLE_GUTSHOT_RE),
        ("open_ended", _OPEN_ENDED_RE),
        ("gutshot", _GUTSHOT_RE),
        ("generic", _STRAIGHT_DRAW_RE),
    )
    for priority, (kind, pattern) in enumerate(specs):
        for draw in pattern.finditer(text, clause_start, clause_end):
            if draw.end() <= outs.start():
                distance = outs.start() - draw.end()
            elif outs.end() <= draw.start():
                distance = draw.start() - outs.end()
            else:
                distance = 0
            candidates.append((distance, priority, kind))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _validate_free_card_street(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Prevent one check from skipping unrevealed betting streets."""

    targets = [
        *[(match, "flop") for match in _FREE_FLOP_RE.finditer(bullet)],
        *[(match, "turn") for match in _FREE_TURN_RE.finditer(bullet)],
        *[(match, "river") for match in _FREE_RIVER_RE.finditer(bullet)],
        *[(match, "showdown") for match in _FREE_SHOWDOWN_RE.finditer(bullet)],
    ]
    for target, destination in sorted(targets, key=lambda item: item[0].start()):
        if _match_is_negated(bullet, target):
            continue
        street = _nearest_street_before(bullet, target.start())
        if street is None:
            destination_label = {
                "flop": "翻牌",
                "turn": "转牌",
                "river": "河牌",
                "showdown": "摊牌",
            }[destination]
            return [
                PokerExperienceIssue(
                    code="free_card_street_undefined",
                    section=section,
                    bullet=bullet,
                    message=(
                        f"声称免费看{destination_label}时必须明确当前行动街道，"
                        "并且一次行动轮只能推进到紧邻的下一街。"
                    ),
                )
            ]
        expected = {
            "flop": "preflop",
            "turn": "flop",
            "river": "turn",
            "showdown": "river",
        }[destination]
        if street != expected:
            return [
                PokerExperienceIssue(
                    code="free_card_street_conflict",
                    section=section,
                    bullet=bullet,
                    message=(
                        "过牌只会结束当前行动轮：翻牌前→翻牌、翻牌→转牌、"
                        "转牌→河牌、河牌→摊牌，不能跳街或停留在同一街。"
                    ),
                )
            ]
    return []


def _validate_free_card_action_chain(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """A free next street requires an explicit hero check in the current action chain."""

    targets = sorted(
        [
            *_FREE_FLOP_RE.finditer(bullet),
            *_FREE_TURN_RE.finditer(bullet),
            *_FREE_RIVER_RE.finditer(bullet),
            *_FREE_SHOWDOWN_RE.finditer(bullet),
        ],
        key=lambda match: match.start(),
    )
    for target in targets:
        if _match_is_negated(bullet, target):
            continue
        clause_start, _ = _clause_bounds(bullet, target.start(), "。；;\n")
        prefix = bullet[clause_start : target.start()]
        if _HERO_CHECK_BEFORE_FREE_CARD_RE.search(prefix):
            continue
        # A BB may receive a walk without checking, but a completed/limped pot
        # reaches the flop only after BB explicitly checks. The gate is limited
        # to prose that actually promises a free next street.
        return [
            PokerExperienceIssue(
                code="free_card_missing_hero_check",
                section=section,
                bullet=bullet,
                message=(
                    "声称免费看下一街或摊牌时必须在同一动作链明确写出我方过牌；"
                    "只写对手过牌、平跟或盲注状态不足以结束本轮行动。"
                ),
            )
        ]
    return []


def _nearest_street_before(text: str, index: int) -> str | None:
    """Return the latest explicit street in the same sentence before ``index``."""

    sentence_start, _ = _clause_bounds(text, index, "。；;\n")
    candidates: list[tuple[int, str]] = []
    for street, pattern in (
        ("preflop", _EXPLICIT_PREFLOP_RE),
        ("flop", _FLOP_CONTEXT_RE),
        ("turn", _TURN_CONTEXT_RE),
        ("river", _RIVER_CONTEXT_RE),
    ):
        candidates.extend(
            (match.start(), street)
            for match in pattern.finditer(text, sentence_start, index)
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _validate_betting_purpose(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Reject information bets and betting described as pot control."""

    issues: list[PokerExperienceIssue] = []
    for fragment_match in re.finditer(r"[^。；;，,\n]+", bullet):
        fragment = fragment_match.group(0)
        fragment_start = fragment_match.start()
        for code, pattern, message in (
            (
                "information_bet_misuse",
                _INFORMATION_BET_RE,
                "下注或加注不能只以获取信息为目标；应说明价值、诈唬、权益拒绝或定价。",
            ),
            (
                "bet_pot_control_misuse",
                _BET_POT_CONTROL_RE,
                "下注会扩大底池，不能把下注或加注本身表述为控池。",
            ),
        ):
            match = pattern.search(fragment)
            if match is None:
                continue
            context = fragment[
                max(0, match.start() - 12) : min(len(fragment), match.end() + 4)
            ]
            if _NEGATION_RE.search(context) or re.search(
                r"而非|不以|不是为了", context
            ):
                continue
            if code == "bet_pot_control_misuse":
                control = _POT_CONTROL_RE.search(
                    bullet,
                    fragment_start + match.start(),
                    fragment_start + match.end(),
                )
                if control is not None and _nearest_street_before(
                    bullet, control.start()
                ) == "river":
                    # The river-specific diagnostic is more precise; do not
                    # emit two codes for the same underlying claim.
                    continue
            issues.append(PokerExperienceIssue(code, section, bullet, message))
    return issues


def _validate_noncommittal_action(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Reject menus or vague caution that cannot guide a concrete decision."""

    generic_menu = _NONCOMMITTAL_ACTION_MENU_RE.search(bullet)
    bare_menu = _BARE_ACTION_MENU_RE.search(bullet)
    direct_menu = _DIRECT_ACTION_MENU_RE.search(bullet)
    vague_caution = _VAGUE_CAUTION_RE.search(bullet)
    offending = generic_menu or vague_caution
    if offending is None and bare_menu is not None:
        tail = bullet[bare_menu.end() :]
        if not _EXPLICIT_MENU_BRANCH_RE.search(tail):
            offending = bare_menu
    if (
        offending is None
        and direct_menu is not None
        and not _LEGAL_ACTION_ENUMERATION_RE.search(bullet)
    ):
        tail = bullet[direct_menu.end() :]
        if not _EXPLICIT_MENU_BRANCH_RE.search(tail):
            offending = direct_menu
    if offending is None:
        return []
    negation_context = bullet[
        max(0, offending.start() - 12) : min(len(bullet), offending.start() + 2)
    ]
    if _NEGATION_RE.search(negation_context):
        return []
    return [
        PokerExperienceIssue(
            code="noncommittal_action_rule",
            section=section,
            bullet=bullet,
            message=(
                "规则必须给出首选动作以及触发反转的可见条件，不能只列动作菜单或"
                "写‘谨慎/避免过度投入’。"
            ),
        )
    ]


def _validate_static_action_legality(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Do not spend learned-experience capacity restating the action guard."""

    if _STATIC_ACTION_LEGALITY_RE.search(bullet) is None:
        return []
    return [
        PokerExperienceIssue(
            code="static_action_legality_rule",
            section=section,
            bullet=bullet,
            message=(
                "长期经验应记录从轨迹中提炼的策略条件，不应仅复述环境已经强制的"
                "合法动作集合；请改写为在可见局面中为何首选某一合法动作。"
            ),
        )
    ]


def _validate_deterministic_opponent_hand_inference(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Observed action changes a range; it never reveals one certain hand class."""

    if _DETERMINISTIC_OPPONENT_HAND_INFERENCE_RE.search(bullet) is None:
        return []
    return [
        PokerExperienceIssue(
            code="deterministic_opponent_hand_inference",
            section=section,
            bullet=bullet,
            message=(
                "公开动作只能使对手范围更偏向某类组合，不能确定地‘说明其有强牌’；"
                "应使用范围和频率语言，并保留诈唬或偏离类型的反转条件。"
            ),
        )
    ]


def _validate_undefined_strategy_threshold(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Reject action thresholds that cannot be evaluated from visible state."""

    vague = _UNDEFINED_STRATEGY_THRESHOLD_RE.search(bullet)
    if vague is None:
        return []
    if _DIRECT_ACTION_MENU_RE.search(bullet) or _NONCOMMITTAL_ACTION_MENU_RE.search(bullet):
        # The action-menu diagnostic already explains the more fundamental
        # inability to choose one action; do not cascade a second vague-word code.
        return []
    clause_start, clause_end = _clause_bounds(bullet, vague.start(), "。；;\n")
    clause = bullet[clause_start:clause_end]
    if _OBSERVABLE_STRATEGY_THRESHOLD_RE.search(clause):
        return []
    return [
        PokerExperienceIssue(
            code="undefined_strategy_threshold",
            section=section,
            bullet=bullet,
            message=(
                "‘合适时机/有利情况下/谨慎行动’本身不是可执行阈值；请写明可见的"
                "牌面、范围、尺度、赔率、筹码或对手频率条件。"
            ),
        )
    ]


def _validate_reversal_branch(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """A labelled reversal must say what the hero does differently."""

    marker = _REVERSAL_MARKER_RE.search(bullet)
    if marker is None:
        return []
    tail = bullet[marker.end() :]
    for pattern in (_HERO_RECOMMENDED_ACTION_RE, _HERO_SIZING_CHANGE_RE):
        for action in pattern.finditer(tail):
            prefix = tail[max(0, action.start() - 12) : action.start()]
            if re.search(r"对手|对方|villain|opponent", prefix, re.IGNORECASE):
                continue
            return []
    return [
        PokerExperienceIssue(
            code="reversal_missing_alternative_action",
            section=section,
            bullet=bullet,
            message=(
                "标注‘反转/例外’时必须写明条件满足后我方改成哪个动作或如何改变"
                "尺度；只列条件不能指导下一次决策。"
            ),
        )
    ]


def _validate_check_showdown_state(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Checking closes action only on the river after the opponent has checked."""

    if _CHECK_FOR_SHOWDOWN_RE.search(bullet) is None:
        return []
    river = _RIVER_CONTEXT_RE.search(bullet) is not None
    opponent_checked = _OPPONENT_ALREADY_CHECKED_RE.search(bullet) is not None
    hero_is_ip = bool(
        _HU_COMBINED_BUTTON_RE.search(bullet)
        or re.search(r"有位置|位置优势|\bIP\b", bullet, re.IGNORECASE)
    )
    if river and opponent_checked and hero_is_ip:
        return []
    return [
        PokerExperienceIssue(
            code="check_showdown_action_state_undefined",
            section=section,
            bullet=bullet,
            message=(
                "翻牌或转牌过牌不能保证直接摊牌；河牌也只有在我方有位置且对手已"
                "过牌时，过牌才会立即结束行动。否则必须写明后续面对下注的分支。"
            ),
        )
    ]


def _validate_conflicting_opponent_archetypes(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Do not merge opponents with opposite continuing tendencies."""

    if _CONFLICTING_CHECK_ARCHETYPES_RE.search(bullet) is None:
        return []
    return [
        PokerExperienceIssue(
            code="opponent_archetype_conflicting_union",
            section=section,
            bullet=bullet,
            message=(
                "check-call 与 check-fold 的继续率和收益来源相反，不能用‘或/和’"
                "合并后共享一个处方；请按可见倾向拆成独立分支。"
            ),
        )
    ]


def _validate_check_fold_value_purpose(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """A check-fold range supplies fold equity, not calls from worse hands."""

    if _CHECK_FOLD_RE.search(bullet) is None or _VALUE_BET_PURPOSE_RE.search(bullet) is None:
        return []
    return [
        PokerExperienceIssue(
            code="value_bet_against_check_fold_purpose_conflict",
            section=section,
            bullet=bullet,
            message=(
                "频繁 check-fold 的范围不会用更差牌支付价值下注；对其下注若有收益，"
                "来源应写成弃牌权益，而不是‘榨取价值’。"
            ),
        )
    ]


def _validate_facing_aggression_price(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Calling or folding to aggression must bind the price/equity state."""

    if (
        _FACING_AGGRESSION_RE.search(bullet) is None
        or _FOLD_OR_CALL_RECOMMENDATION_RE.search(bullet) is None
        or _HERO_MARGINAL_HAND_RE.search(bullet) is None
        or _AGGRESSION_PRICE_OR_EQUITY_RE.search(bullet) is not None
    ):
        return []
    return [
        PokerExperienceIssue(
            code="facing_aggression_missing_price",
            section=section,
            bullet=bullet,
            message=(
                "面对下注或加注时，跟注/弃牌处方必须绑定可见价格：下注尺度、"
                "底池赔率、所需权益、SPR 或有效筹码；不能忽略价格机械执行。"
            ),
        )
    ]


def _validate_postflop_check_state(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """A postflop check prescription must establish that checking is available."""

    if section.strip() != "注码与位置":
        return []
    if (
        _POSTFLOP_CONTEXT_RE.search(bullet) is None
        or _HERO_CHECK_RECOMMENDATION_RE.search(bullet) is None
        or _CHECK_AVAILABLE_STATE_RE.search(bullet) is not None
        or _OPPONENT_ALREADY_CHECKED_RE.search(bullet) is not None
        or _DIRECT_ACTION_MENU_RE.search(bullet) is not None
        or _NONCOMMITTAL_ACTION_MENU_RE.search(bullet) is not None
    ):
        return []
    return [
        PokerExperienceIssue(
            code="postflop_check_missing_action_state",
            section=section,
            bullet=bullet,
            message=(
                "翻后推荐过牌必须先说明我方是当前首位行动者，或对手已经过牌到我方；"
                "仅凭牌力、位置或底池大小不能证明过牌在当前动作状态可用。"
            ),
        )
    ]


def _validate_static_invariant_experience(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Pure table-order facts belong in the environment, not learned memory."""

    static_order = bool(
        (_TWO_PLAYER_TABLE_RE.search(bullet) or _HU_COMBINED_BUTTON_RE.search(bullet))
        and re.search(r"翻牌前[^。；;\n]{0,16}先行动", bullet)
        and re.search(r"翻牌后[^。；;\n]{0,16}后(?:于|行动)", bullet)
        and re.search(r"(?:BB|大盲位?)[^。；;\n]{0,20}翻牌后[^。；;\n]{0,16}先行动", bullet)
    )
    if not static_order:
        return []
    return [
        PokerExperienceIssue(
            code="static_invariant_not_experience",
            section=section,
            bullet=bullet,
            message=(
                "这条只复述 HU 固定行动顺序和位置事实，未记录由轨迹提炼的局面处方；"
                "静态规则应由环境保证，不占用经验文档容量。"
            ),
        )
    ]


def _validate_check_call_state(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """A check-call line must bind hero first action and the intervening bet."""

    if _CHECK_CALL_LINE_RE.search(bullet) is None or _CHECK_CALL_STATE_RE.search(bullet):
        return []
    return [
        PokerExperienceIssue(
            code="check_call_action_state_undefined",
            section=section,
            bullet=bullet,
            message=(
                "check-call 是连续动作线：我方必须先处于无位置并过牌，随后面对对手"
                "下注才可跟注；请写明位置、下注尺度或所需权益，不能把它当单一步动作。"
            ),
        )
    ]


def _validate_ambiguous_poker_terms(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Require established hand/decision terminology in the persistent rulebook."""

    ambiguous = _AMBIGUOUS_POKER_TERM_RE.search(bullet)
    if ambiguous is None or _match_is_negated(bullet, ambiguous):
        return []
    if ambiguous.group("high_pair"):
        message = "‘高对’含义不明确；请按事实写成顶对、超对或具体已成牌类别。"
    else:
        message = "‘支付坚果’不是完整策略术语；请写成避免用次坚果向坚果范围支付。"
    return [
        PokerExperienceIssue(
            code="ambiguous_poker_term",
            section=section,
            bullet=bullet,
            message=message,
        )
    ]


def _validate_hu_isolation(section: str, bullet: str) -> list[PokerExperienceIssue]:
    """Isolation is not a meaningful objective when BTN/SB already faces one HU opponent."""

    for clause in re.split(r"[。；;\n]", bullet):
        isolation = _HU_SINGLE_OPPONENT_PURPOSE_RE.search(clause)
        if isolation is None or _match_is_negated(clause, isolation):
            continue
        if _HU_COMBINED_BUTTON_RE.search(clause) or _TWO_PLAYER_TABLE_RE.search(
            clause
        ):
            return [
                PokerExperienceIssue(
                    code="hu_isolation_misuse",
                    section=section,
                    bullet=bullet,
                    message=(
                        "两人桌已经只有一个对手；加注目的应写价值建池、诈唬或"
                        "施压，而不是隔离或争取单挑。"
                    ),
                )
            ]
    return []


def _validate_ambiguous_hand_terms(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Reject fused board/hero labels that do not identify the made hand."""

    if not _AMBIGUOUS_PAIRED_HIGH_CARD_RE.search(bullet):
        return []
    return [
        PokerExperienceIssue(
            code="ambiguous_paired_high_card_term",
            section=section,
            bullet=bullet,
            message=(
                "“成对K高牌”等写法混淆牌面结构与我方成牌；应明确写成"
                "“牌面成对且我方仅K高”或“我方持有一对K”。"
            ),
        )
    ]


def _validate_suit_suffixes(section: str, bullet: str) -> list[PokerExperienceIssue]:
    hands = list(_SUFFIXED_HAND_RE.finditer(bullet))
    if not hands:
        return []
    issues: list[PokerExperienceIssue] = [
        PokerExperienceIssue(
            code="pair_suitedness_suffix_invalid",
            section=section,
            bullet=bullet,
            message=(
                f"{hand.group(0)} 是口袋对子；对子不存在 suited/offsuit 后缀之分。"
            ),
        )
        for hand in hands
        if hand.group("high").upper() == hand.group("low").upper()
    ]
    for term, stated_suffix in [
        *[(match, "o") for match in _OFFSUIT_TERM_RE.finditer(bullet)],
        *[(match, "s") for match in _SUITED_TERM_RE.finditer(bullet)],
    ]:
        clause_start, clause_end = _clause_bounds(bullet, term.start(), "。；;\n")
        clause_hands = [
            hand
            for hand in hands
            if clause_start <= hand.start() and hand.end() <= clause_end
        ]
        hand = _nearest_hand(term, clause_hands, bullet, max_distance=14)
        if hand is None:
            continue
        key = (hand.group("high").upper(), hand.group("low").upper())
        same_hand_variants = [
            candidate
            for candidate in clause_hands
            if (
                candidate.group("high").upper(),
                candidate.group("low").upper(),
            )
            == key
        ]
        clause_variants = {
            candidate.group("suffix").lower() for candidate in same_hand_variants
        }
        if len(clause_variants) > 1:
            clause = bullet[clause_start:clause_end]
            if _COLLECTIVE_SUIT_ASSERTION_RE.search(clause):
                candidates = same_hand_variants
            else:
                between = bullet[
                    min(hand.end(), term.end()) : max(hand.start(), term.start())
                ]
                if not _DIRECT_LABEL_ASSERTION_RE.search(between):
                    # A comparison such as "compare AKs with AKo; the suited
                    # version..." is not asserting the label for both variants.
                    continue
                candidates = [hand]
        else:
            candidates = [hand]
        for candidate in candidates:
            if _claim_is_negated(bullet, candidate, term):
                continue
            actual_suffix = candidate.group("suffix").lower()
            if actual_suffix == stated_suffix:
                continue
            token = candidate.group(0)
            issues.append(
                PokerExperienceIssue(
                    code="hand_suitedness_conflict",
                    section=section,
                    bullet=bullet,
                    message=(
                        f"{token} 的后缀表示"
                        f"{'同花' if actual_suffix == 's' else '非同花'}，"
                        "但相邻文字给出了相反表述。"
                    ),
                )
            )
    return issues


def _validate_connectivity_labels(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    hands = list(_RANKED_HAND_RE.finditer(bullet))
    if not hands:
        return []
    issues: list[PokerExperienceIssue] = []
    labeled_terms = [
        *[(match, "connector") for match in _CONNECTOR_TERM_RE.finditer(bullet)],
        *[(match, "gapper") for match in _GAPPER_TERM_RE.finditer(bullet)],
    ]
    for term, label in labeled_terms:
        hand = _nearest_hand(term, hands, bullet, max_distance=14)
        if hand is None:
            continue
        clause_start, clause_end = _clause_bounds(bullet, term.start(), "。；;\n")
        clause = bullet[clause_start:clause_end]
        between = bullet[min(hand.end(), term.end()) : max(hand.start(), term.start())]
        if _COMPARISON_LABEL_RE.search(clause) and not _DIRECT_LABEL_ASSERTION_RE.search(
            between
        ):
            continue
        if _claim_is_negated(bullet, hand, term):
            continue
        local_start = max(0, min(hand.start(), term.start()) - 4)
        local_end = min(len(bullet), max(hand.end(), term.end()) + 2)
        if label == "connector" and _CONNECTIVITY_NEGATION_RE.search(
            bullet[local_start:local_end]
        ):
            continue
        high = hand.group("high").upper()
        low = hand.group("low").upper()
        if {high, low} == {"A", "2"}:
            # A2 is described inconsistently across poker vocabularies; stay conservative.
            continue
        gap = abs(_RANK_VALUE[high] - _RANK_VALUE[low])
        token = hand.group(0)
        if label == "gapper" and gap == 1:
            issues.append(
                PokerExperienceIssue(
                    code="connector_called_gapper",
                    section=section,
                    bullet=bullet,
                    message=f"{token} 的两个点数相邻，属于连接牌，不应称为间隔牌。",
                )
            )
        elif label == "connector" and gap != 1:
            issues.append(
                PokerExperienceIssue(
                    code="nonconnector_called_connector",
                    section=section,
                    bullet=bullet,
                    message=f"{token} 的两个点数不相邻，不应称为连接牌。",
                )
            )
    return issues


def _validate_hu_postflop_position(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    # BTN is last and BB is out of position postflop in ordinary Hold'em,
    # including heads-up. Requiring the literal word "HU" allowed rules such
    # as "BB 利用位置看翻牌" to escape validation.
    if not _POSTFLOP_RE.search(bullet):
        return []
    issues: list[PokerExperienceIssue] = []
    role_specs = (
        (
            _BTN_RE,
            _BTN_WRONG_RE,
            "hu_btn_postflop_position_conflict",
            "单挑牌局中 BTN 翻后后行动并拥有位置。",
        ),
        (
            _BB_RE,
            _BB_WRONG_RE,
            "hu_bb_postflop_position_conflict",
            "单挑牌局中 BB 翻后先行动并处于无位置。",
        ),
    )
    all_roles = [*_BTN_RE.finditer(bullet), *_BB_RE.finditer(bullet)]
    for role_re, wrong_re, code, message in role_specs:
        for role in role_re.finditer(bullet):
            # A heads-up *pot* is not necessarily a two-player table.  On a
            # multi-player table after everyone else folds, BB acts after SB
            # postflop.  Only the explicit two-player-table case makes this
            # BB-vs-SB exception invalid (the HU BTN is also the SB).
            if (
                role_re is _BB_RE
                and _BB_VS_SB_RE.search(bullet)
                and not _TWO_PLAYER_TABLE_RE.search(bullet)
            ):
                continue
            claim = _nearest_direct_claim(role, wrong_re, all_roles, bullet)
            if (
                claim is None
                or _position_claim_is_negated(bullet, role, claim)
                or _claim_is_preflop_scoped(bullet, claim)
            ):
                continue
            issues.append(PokerExperienceIssue(code, section, bullet, message))
            break
    if _TWO_PLAYER_TABLE_RE.search(bullet):
        hu_roles = [
            *_BTN_RE.finditer(bullet),
            *_BB_RE.finditer(bullet),
            *_SB_RE.finditer(bullet),
        ]
        for role in _SB_RE.finditer(bullet):
            claim = _nearest_direct_claim(role, _BTN_WRONG_RE, hu_roles, bullet)
            if (
                claim is None
                or _position_claim_is_negated(bullet, role, claim)
                or _claim_is_preflop_scoped(bullet, claim)
            ):
                continue
            issues.append(
                PokerExperienceIssue(
                    "hu_sb_postflop_position_conflict",
                    section,
                    bullet,
                    "两人桌中 SB 同时是 BTN，翻后后行动并拥有位置。",
                )
            )
            break
        # Long Chinese clauses can place the street after the position claim,
        # for example ``HU 中 BB ... 利用位置优势在翻牌后决策``.  Re-run the
        # same role-bound matcher with a wider same-clause window.  Do not use
        # a raw regex search: that would bind ``BB 先行动，BTN 最后行动`` to
        # the wrong role.
        existing_codes = {issue.code for issue in issues}
        for clause in re.split(r"[。；;\n]", bullet):
            if not _EXPLICIT_POSTFLOP_RE.search(clause):
                continue
            clause_roles = [*_BTN_RE.finditer(clause), *_BB_RE.finditer(clause)]
            for role_re, wrong_re, code, message in role_specs:
                if code in existing_codes:
                    continue
                for role in role_re.finditer(clause):
                    claim = _nearest_direct_claim(
                        role,
                        wrong_re,
                        clause_roles,
                        clause,
                        max_distance=96,
                    )
                    if claim is None or _claim_is_preflop_scoped(clause, claim):
                        continue
                    if _position_claim_is_negated(clause, role, claim):
                        continue
                    issues.append(PokerExperienceIssue(code, section, bullet, message))
                    existing_codes.add(code)
                    break
    return issues


def _validate_hu_ip_check_order(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """A HU BTN/SB check-back is legal only after the OOP opponent checks."""

    for clause in re.split(r"[。；;\n]", bullet):
        button_roles = list(_HU_COMBINED_BUTTON_RE.finditer(clause))
        if not button_roles:
            continue
        if not _EXPLICIT_POSTFLOP_RE.search(clause):
            continue
        hero_check = re.search(
            r"(?:我方|我们|自己|本人|hero).{0,6}过牌|"
            r"(?:可|应|选择|继续)\s*过牌|"
            r"(?:可|应|选择|继续)[^。；;\n]{0,18}"
            r"(?:下注\s*(?:或|/)\s*过牌|过牌\s*(?:或|/)\s*下注)",
            clause,
            re.IGNORECASE,
        )
        if hero_check is None or _match_is_negated(clause, hero_check):
            continue
        # Bind the action to BTN/SB rather than merely requiring that the role
        # appears somewhere in the sentence.  In prose such as ``在BB位置，翻牌前
        # 面对BTN/SB的complete，应过牌……但翻后……``, BTN/SB names the
        # opponent; the check belongs to BB and is not a check-back claim.
        acting_button_roles = [
            role
            for role in button_roles
            if role.end() <= hero_check.start()
            and not re.search(
                r"(?:面对|对阵|对抗|迎战|versus|\bvs\.?)[^，,。；;]{0,6}$",
                clause[max(0, role.start() - 16) : role.start()],
                re.IGNORECASE,
            )
        ]
        if not acting_button_roles:
            continue
        prior = clause[: hero_check.start()]
        if _OPPONENT_PRIOR_CHECK_RE.search(prior):
            continue
        return [
            PokerExperienceIssue(
                code="hu_ip_check_missing_prior_check",
                section=section,
                bullet=bullet,
                message=(
                    "两人桌 BTN/SB 翻后有位置；其过牌是 check-back，必须先写明"
                    "无位置对手已经过牌，不能省略先行动者。"
                ),
            )
        ]
    return []


def _validate_hu_bb_opponent_check_raise(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """The in-position HU opponent cannot check before an out-of-position BB."""

    if not (_TWO_PLAYER_TABLE_RE.search(bullet) or _HU_COMBINED_BUTTON_RE.search(bullet)):
        return []
    for clause in re.split(r"[。；;\n]", bullet):
        conflict = _HU_BB_OPPONENT_CHECK_RAISE_RE.search(clause)
        if conflict is None or _match_is_negated(clause, conflict):
            continue
        prior = bullet[: bullet.find(clause) + conflict.start()]
        # Require the document to have established BB as the hero/acting
        # position.  ``BTN 面对 BB 的 check-raise`` is a valid HU line.
        bb_role = list(_BB_RE.finditer(prior))
        button_role = list(_HU_COMBINED_BUTTON_RE.finditer(prior))
        if not bb_role:
            continue
        if button_role and button_role[-1].start() > bb_role[-1].start():
            continue
        return [
            PokerExperienceIssue(
                code="hu_bb_opponent_check_raise_impossible",
                section=section,
                bullet=bullet,
                message=(
                    "两人桌 BB 翻后先行动；当我方是 BB 时，后行动的 BTN/SB 不可能"
                    "先过牌再对我方 check-raise。应改写为面对其加注，或改正我方位置。"
                ),
            )
        ]
    return []


def _validate_ambiguous_paired_state(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Require pair statements to name hero's hand or the public board."""

    for state in _AMBIGUOUS_PAIRED_STATE_RE.finditer(bullet):
        if _match_is_negated(bullet, state):
            continue
        clause_start, clause_end = _clause_bounds(bullet, state.start(), "。；;\n")
        clause = bullet[clause_start:clause_end]
        if _HERO_OR_BOARD_PAIR_SUBJECT_RE.search(clause):
            continue
        local_state_start = state.start() - clause_start
        if _OPPONENT_SUBJECT_RE.search(clause[max(0, local_state_start - 12) : local_state_start]):
            continue
        return [
            PokerExperienceIssue(
                code="paired_state_ambiguous",
                section=section,
                bullet=bullet,
                message=(
                    "“成对/持有对子”必须明确主语：写成牌面成对，或我方底牌"
                    "形成一对；不能把公共牌结构与我方牌力混为一谈。"
                ),
            )
        ]
    return []


def _validate_board_threat_draw_subject(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Do not silently turn a board texture threat into hero's own draw."""

    threat = _BOARD_DRAW_THREAT_RE.search(bullet)
    result = _ANONYMOUS_DRAW_RESULT_RE.search(bullet)
    if threat is None or result is None or result.start() <= threat.start():
        return []
    if _match_is_negated(bullet, result) or _EXPLICIT_HERO_DRAW_RE.search(bullet):
        return []
    return [
        PokerExperienceIssue(
            code="board_threat_recast_as_hero_draw",
            section=section,
            bullet=bullet,
            message=(
                "牌面的同花/顺子威胁不等于我方持有听牌；后续写“未完成听牌”前"
                "必须用我方底牌明确绑定听牌主语。"
            ),
        )
    ]


def _validate_paired_board_two_pair(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    if not _PAIRED_BOARD_RE.search(bullet):
        return []
    claims = [
        *_PAIRED_BOARD_CAUSAL_TWO_PAIR_RE.finditer(bullet),
        *_TWO_PAIR_FROM_PAIRED_BOARD_RE.finditer(bullet),
    ]
    for claim in claims:
        clause = _claim_clause(bullet, claim)
        if _match_is_negated(bullet, claim) or _match_is_range_statement(bullet, claim):
            continue
        if _DOUBLE_PAIRED_BOARD_RE.search(clause):
            continue
        if _has_hero_bound_condition(clause, _HOLE_PAIR_CONDITION_RE):
            continue
        if _has_hero_bound_condition(clause, _EXPLICIT_TWO_PAIR_STATE_RE):
            continue
        return [
            PokerExperienceIssue(
                code="paired_board_two_pair_without_hole_condition",
                section=section,
                bullet=bullet,
                message=(
                    "单一成对公共牌本身只提供一对；称为两对时必须写明底牌形成"
                    "另一对等额外条件。"
                ),
            )
        ]
    return []


def _validate_river_pot_control(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    for control in _POT_CONTROL_RE.finditer(bullet):
        if _nearest_street_before(bullet, control.start()) != "river":
            continue
        # A purpose clause after the phrase (for example, "控池，避免...")
        # does not negate the pot-control claim itself. Inspect only its left
        # context so that the purpose does not hide the misuse.
        context = bullet[max(0, control.start() - 12) : control.end()]
        if _NEGATION_RE.search(context):
            continue
        return [
            PokerExperienceIssue(
                code="river_pot_control_misuse",
                section=section,
                bullet=bullet,
                message=(
                    "河牌后已无后续街道；应描述价值下注、诈唬、过牌摊牌等目的，"
                    "而不是把动作理由写成控池。"
                ),
            )
        ]
    return []


def _validate_river_future_draw(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """River decisions compare made-hand ranges; no future card remains to make a draw."""

    misuse = _RIVER_FUTURE_DRAW_RE.search(bullet)
    if misuse is None:
        return []
    return [
        PokerExperienceIssue(
            code="river_future_draw_misuse",
            section=section,
            bullet=bullet,
            message=(
                "河牌后不再有未来牌可发，不能用“更高听牌可能”判断价值；应比较"
                "对手可能持有的更高已成牌、坚果组合和阻断牌。"
            ),
        )
    ]


def _validate_postflop_raise_requires_facing_action(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """A postflop raise requires a live opponent bet; checked-to action is a bet."""

    active_postflop = section.strip() in {"翻牌后", "转牌 / 河牌"}
    for clause in re.split(r"[。；;\n]", bullet):
        if _HAND_RANK_STRONGER_RE.search(clause) or _HAND_RANK_WEAKER_RE.search(clause):
            # A deliberately invalid fixed-ranking sentence may contain an
            # action word as its example.  Emit the more fundamental ranking
            # diagnostic and avoid a cascading action-state error.
            continue
        preflop_matches = list(_EXPLICIT_PREFLOP_RE.finditer(clause))
        postflop_matches = list(_EXPLICIT_POSTFLOP_RE.finditer(clause))
        if section.strip() == "起手牌" and not postflop_matches:
            active_postflop = False
        if preflop_matches or postflop_matches:
            latest_preflop = preflop_matches[-1].start() if preflop_matches else -1
            latest_postflop = postflop_matches[-1].start() if postflop_matches else -1
            active_postflop = latest_postflop >= latest_preflop
        if not active_postflop:
            continue
        recommendation = _HERO_RAISE_RECOMMENDATION_RE.search(clause)
        if recommendation is None or _match_is_negated(clause, recommendation):
            continue
        recommendation_text = recommendation.group(0)
        raise_offset = max(
            recommendation_text.rfind("加注"),
            recommendation_text.lower().rfind("raise"),
        )
        raise_prefix = recommendation_text[max(0, raise_offset - 8) : raise_offset]
        # “可跟注翻牌前加注” recommends a call *against* a raise.  The broad
        # recommendation regex deliberately captures enough context to find the
        # word 加注, so disambiguate an earlier hero response verb before treating
        # that final token as the recommended action.
        response_prefix = recommendation_text[:raise_offset]
        if re.search(r"(?:跟注|平跟|call)[^。；;\n]{0,12}$", response_prefix, re.IGNORECASE):
            continue
        if re.search(r"而非|并非|不是|不应|不要|不能|不可", raise_prefix):
            continue
        if re.search(r"(?:对手|对方|其)[^。；;\n]{0,8}$", raise_prefix):
            continue
        prior_postflop_markers = [
            match for match in postflop_matches if match.end() <= recommendation.start()
        ]
        action_context_start = (
            prior_postflop_markers[-1].end() if prior_postflop_markers else 0
        )
        action_context = clause[action_context_start : recommendation.end()]
        if _OPPONENT_BET_OR_RAISE_RE.search(action_context):
            continue
        return [
            PokerExperienceIssue(
                code="postflop_raise_without_facing_action",
                section=section,
                bullet=bullet,
                message=(
                    "翻后加注必须先面对对手的下注或加注；若行动过牌到我方，"
                    "主动投入应写下注而不是加注。"
                ),
            )
        ]
    return []


def _validate_board_made_hand_claims(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    issues: list[PokerExperienceIssue] = []
    flush_specs = (
        (_AUTO_THREE_SUIT_FLUSH_RE, _THREE_SUIT_HOLE_CONDITION_RE),
        (_AUTO_FOUR_SUIT_FLUSH_RE, _FOUR_SUIT_HOLE_CONDITION_RE),
    )
    for claim_re, condition_re in flush_specs:
        if any(
            not _match_is_negated(bullet, claim)
            and not _match_is_range_statement(bullet, claim)
            and not _has_hero_bound_condition(
                _claim_clause(bullet, claim), condition_re
            )
            for claim in claim_re.finditer(bullet)
        ):
            issues.append(
                PokerExperienceIssue(
                    code="suited_board_auto_flush_without_hole_condition",
                    section=section,
                    bullet=bullet,
                    message=(
                        "三张或四张同花公共牌不会自动让任意底牌成为同花；"
                        "必须写明足够数量的同花底牌或已成同花条件。"
                    ),
                )
            )
            break
    if any(
        not _match_is_negated(bullet, claim)
        and not _match_is_range_statement(bullet, claim)
        and not _has_hero_bound_condition(
            _claim_clause(bullet, claim), _HOLE_STRAIGHT_CONDITION_RE
        )
        for claim in _AUTO_STRAIGHT_FROM_BOARD_RE.finditer(bullet)
    ):
        issues.append(
            PokerExperienceIssue(
                code="connected_board_auto_straight_without_hole_condition",
                section=section,
                bullet=bullet,
                message=(
                    "四连公共牌不会自动让任意底牌成为顺子；"
                    "必须写明底牌补足的点数或已成顺子条件。"
                ),
            )
        )
    if any(
        not _match_is_negated(bullet, claim)
        and not _match_is_range_statement(bullet, claim)
        and not _has_hero_bound_condition(
            _claim_clause(bullet, claim), _HOLE_MATCHES_PAIRED_RANK_RE
        )
        for claim in _PAIRED_BOARD_AUTO_TRIPS_RE.finditer(bullet)
    ):
        issues.append(
            PokerExperienceIssue(
                code="paired_board_auto_stronger_hand_without_hole_condition",
                section=section,
                bullet=bullet,
                message=(
                    "单一成对公共牌不会自动让任意底牌成为三条；"
                    "必须写明底牌命中牌面对子的条件。"
                ),
            )
        )
    if any(
        not _match_is_negated(bullet, claim)
        and not _match_is_range_statement(bullet, claim)
        and not _has_hero_bound_condition(
            _claim_clause(bullet, claim), _EXPLICIT_FULL_HOUSE_STATE_RE
        )
        for claim in _PAIRED_BOARD_AUTO_FULL_HOUSE_RE.finditer(bullet)
    ):
        issues.append(
            PokerExperienceIssue(
                code="paired_board_auto_stronger_hand_without_hole_condition",
                section=section,
                bullet=bullet,
                message=(
                    "单一成对公共牌不会自动让任意底牌成为葫芦；"
                    "必须把我方已成葫芦写成独立可见条件。"
                ),
            )
        )
    return issues


def _validate_hand_ranking(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    issues: list[PokerExperienceIssue] = []
    comparisons = (
        (_HAND_RANK_STRONGER_RE, True),
        (_HAND_RANK_WEAKER_RE, False),
    )
    for comparison_re, claims_stronger in comparisons:
        for claim in comparison_re.finditer(bullet):
            if _match_is_negated(bullet, claim):
                continue
            left = claim.group("left")
            right = claim.group("right")
            left_rank = _HAND_CLASS_RANK[left]
            right_rank = _HAND_CLASS_RANK[right]
            valid = left_rank > right_rank if claims_stronger else left_rank < right_rank
            if valid:
                continue
            issues.append(
                PokerExperienceIssue(
                    code="hand_ranking_conflict",
                    section=section,
                    bullet=bullet,
                    message=f"固定牌型等级不支持“{claim.group(0)}”这一比较。",
                )
            )
    return issues


def _claim_clause(text: str, claim: re.Match[str]) -> str:
    start, end = _clause_bounds(text, claim.start(), "。；;\n")
    return text[start:end]


def _match_is_negated(text: str, claim: re.Match[str]) -> bool:
    context = text[max(0, claim.start() - 10) : claim.end()]
    return bool(_NEGATION_RE.search(context))


def _match_is_range_statement(text: str, claim: re.Match[str]) -> bool:
    if _OPPONENT_SUBJECT_RE.search(claim.group(0)):
        return True
    tail = text[claim.end() : min(len(text), claim.end() + 16)]
    return bool(_RANGE_MADE_HAND_SUFFIX_RE.search(tail))


def _has_hero_bound_condition(
    clause: str, condition_re: re.Pattern[str]
) -> bool:
    """Accept an implicit-hero condition unless nearby text assigns it to villain."""

    for condition in condition_re.finditer(clause):
        if _HERO_SUBJECT_RE.search(condition.group(0)):
            return True
        left_context = clause[max(0, condition.start() - 16) : condition.start()]
        hero_subjects = list(_HERO_SUBJECT_RE.finditer(left_context))
        opponent_subjects = list(_OPPONENT_SUBJECT_RE.finditer(left_context))
        hero_position = hero_subjects[-1].start() if hero_subjects else -1
        opponent_position = opponent_subjects[-1].start() if opponent_subjects else -1
        if opponent_position > hero_position:
            continue
        return True
    return False


def _validate_reraise_chain(section: str, bullet: str) -> list[PokerExperienceIssue]:
    issues: list[PokerExperienceIssue] = []
    for reraised in _OPPONENT_RERAISE_RE.finditer(bullet):
        context = bullet[max(0, reraised.start() - 8) : reraised.end()]
        if _NEGATION_RE.search(context):
            continue
        # Keep semicolon-separated actions in one sequence: poker histories
        # commonly write "opponent raises; hero 3-bets; opponent re-raises".
        sentence_start = max(
            bullet.rfind("。", 0, reraised.start()),
            bullet.rfind("\n", 0, reraised.start()),
        )
        street_matches = list(
            _STREET_MARKER_RE.finditer(
                bullet, sentence_start + 1, reraised.start()
            )
        )
        sequence_start = (
            street_matches[-1].start() if street_matches else sentence_start + 1
        )
        prefix = bullet[sequence_start: reraised.start()]
        if _HERO_RAISE_RE.search(prefix) or _THIRD_PARTY_RAISE_RE.search(prefix):
            continue
        issues.append(
            PokerExperienceIssue(
                code="opponent_reraise_missing_intermediate_raise",
                section=section,
                bullet=bullet,
                message=(
                    "“对手再次加注”需要在动作链中先出现我方或第三方的中间加注；"
                    "当前 bullet 未交代该动作。"
                ),
            )
        )
    return issues


def _validate_preflop_section_placement(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Keep concrete preflop prescriptions in the starting-hand section."""

    if section.strip() != "注码与位置":
        return []
    if not _PREFLOP_CONTEXT_RE.search(bullet):
        return []
    action = _PREFLOP_SPECIFIC_ACTION_RE.search(bullet)
    if action is None or _match_is_negated(bullet, action):
        return []
    if not (
        _RANKED_HAND_RE.search(bullet)
        or _HYPHENATED_HOLE_CARD_RE.search(bullet)
        or _SPECIFIC_STARTING_HAND_RE.search(bullet)
    ):
        return []
    return [
        PokerExperienceIssue(
            code="misplaced_preflop_action_rule",
            section=section,
            bullet=bullet,
            message=(
                "具体翻前跟注、弃牌、加注或过牌处方属于“起手牌”；"
                "“注码与位置”只保留不重复具体动作的筹码深度、尺度与相对位置原则。"
            ),
        )
    ]


def _validate_later_street_section_placement(
    section: str, bullet: str
) -> list[PokerExperienceIssue]:
    """Keep a turn/river recommendation out of the flop chapter.

    Merely mentioning the flop as historical setup does not make the rule a
    flop decision.  Bind the street to the recommended hero action instead.
    """

    if section.strip() != "翻牌后":
        return []
    action_streets = _recommended_action_streets(bullet)
    if not action_streets or action_streets.intersection({"flop", "unspecified"}):
        return []
    if not action_streets.issubset({"turn", "river"}):
        return []
    return [
        PokerExperienceIssue(
            code="misplaced_later_street_rule",
            section=section,
            bullet=bullet,
            message=(
                "只描述转牌或河牌动作的规则属于“转牌 / 河牌”；“翻牌后”章节"
                "应保留翻牌圈处方或明确跨街条件链。"
            ),
        )
    ]


def recommended_action_streets(bullet: str) -> frozenset[str]:
    """Public helper used by the document-level chapter coverage gate."""

    return frozenset(_recommended_action_streets(bullet))


def _recommended_action_streets(bullet: str) -> set[str]:
    streets: set[str] = set()
    street_patterns = (
        ("preflop", _EXPLICIT_PREFLOP_RE),
        ("flop", _FLOP_CONTEXT_RE),
        ("turn", _TURN_CONTEXT_RE),
        ("river", _RIVER_CONTEXT_RE),
    )
    for clause in re.split(r"[。；;\n]", bullet):
        markers = sorted(
            (
                (match.start(), label)
                for label, pattern in street_patterns
                for match in pattern.finditer(clause)
            ),
            key=lambda item: item[0],
        )
        for recommendation in _HERO_RECOMMENDED_ACTION_RE.finditer(clause):
            subject_prefix = clause[
                max(0, recommendation.start() - 12) : recommendation.start()
            ]
            if re.search(
                r"(?:对手|对方|其|villain|opponent)[^，,。；;\n]{0,5}$",
                subject_prefix,
                re.IGNORECASE,
            ):
                continue
            action_position = recommendation.start("action")
            preceding = [item for item in markers if item[0] <= action_position]
            if preceding:
                streets.add(preceding[-1][1])
            else:
                streets.add("unspecified")
        for marker_position, label in markers:
            tail = clause[marker_position : marker_position + 24]
            direct_action = re.search(r"弃牌|过牌|跟注|加注|下注", tail)
            if direct_action is None:
                continue
            between = tail[: direct_action.start()]
            if re.search(r"对手|对方|其|双方|villain|opponent", between, re.IGNORECASE):
                continue
            if _NEGATION_RE.search(between):
                continue
            streets.add(label)
    return streets


def _clause_bounds(text: str, index: int, separators: str) -> tuple[int, int]:
    left = max((text.rfind(separator, 0, index) for separator in separators), default=-1)
    right_candidates = [
        position
        for separator in separators
        for position in [text.find(separator, index)]
        if position >= 0
    ]
    return left + 1, min(right_candidates, default=len(text))


def _nearest_hand(
    term: re.Match[str],
    hands: list[re.Match[str]],
    text: str,
    *,
    max_distance: int,
) -> re.Match[str] | None:
    candidates: list[tuple[int, re.Match[str]]] = []
    for hand in hands:
        if hand.end() <= term.start():
            distance = term.start() - hand.end()
            between = text[hand.end() : term.start()]
        elif term.end() <= hand.start():
            distance = hand.start() - term.end()
            between = text[term.end() : hand.start()]
        else:
            distance = 0
            between = ""
        if (
            distance <= max_distance
            and not _CLAUSE_BREAK_RE.search(between)
            and not _HAND_LABEL_BINDING_BARRIER_RE.search(between)
        ):
            candidates.append((distance, hand))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].start()))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1]


def _nearest_direct_claim(
    role: re.Match[str],
    claim_re: re.Pattern[str],
    all_roles: list[re.Match[str]],
    text: str,
    *,
    max_distance: int = 12,
) -> re.Match[str] | None:
    candidates: list[tuple[int, re.Match[str]]] = []
    for claim in claim_re.finditer(text):
        # Chinese strategic prose normally places the acting role before its
        # predicate ("BTN 先行动"). Accepting a predicate before the role can
        # accidentally bind "BB 先行动，BTN ..." to BTN, so fail closed here.
        if role.end() > claim.start():
            continue
        distance = claim.start() - role.end()
        between = text[role.end() : claim.start()]
        if distance > max_distance or _CLAUSE_BREAK_RE.search(between) or "相对" in between:
            continue
        if any(
            other is not role
            and min(role.end(), claim.end()) <= other.start() < max(role.start(), claim.start())
            for other in all_roles
        ):
            continue
        candidates.append((distance, claim))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].start()))
    return candidates[0][1]


def _claim_is_negated(
    text: str, role: re.Match[str], claim: re.Match[str]
) -> bool:
    start = max(0, min(role.start(), claim.start()) - 8)
    end = min(len(text), max(role.end(), claim.end()) + 2)
    return bool(_NEGATION_RE.search(text[start:end]))


def _position_claim_is_negated(
    text: str, role: re.Match[str], claim: re.Match[str]
) -> bool:
    """Recognize explicit polarity denial without weakening global negation rules."""

    if _claim_is_negated(text, role, claim):
        return True
    if claim.group(0).lower() == "无位置":
        # Bare ``无位置`` is itself the wrong BTN/SB postflop claim, not a
        # negation marker.  ``无位置劣势`` explicitly denies a disadvantage
        # and remains valid; an external denial such as ``并非无位置`` was
        # already handled by _claim_is_negated above.
        return text[claim.end() : claim.end() + 2] == "劣势"
    start = max(0, min(role.start(), claim.start()) - 8)
    end = min(len(text), max(role.end(), claim.end()) + 2)
    return bool(_POSITION_POLARITY_NEGATION_RE.search(text[start:end]))


def _claim_is_preflop_scoped(text: str, claim: re.Match[str]) -> bool:
    """Return whether the nearest explicit street before a position claim is preflop."""

    context = text[max(0, claim.start() - 24) : claim.start()]
    preflop = list(_EXPLICIT_PREFLOP_RE.finditer(context))
    if not preflop:
        return False
    postflop = list(_EXPLICIT_POSTFLOP_RE.finditer(context))
    return not postflop or preflop[-1].start() > postflop[-1].start()


def _deduplicate_issues(
    issues: list[PokerExperienceIssue],
) -> list[PokerExperienceIssue]:
    seen: set[tuple[str, str, str]] = set()
    result: list[PokerExperienceIssue] = []
    for issue in issues:
        key = (issue.code, issue.section, issue.bullet)
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result


__all__ = ["PokerExperienceIssue", "validate_poker_experience_document"]
