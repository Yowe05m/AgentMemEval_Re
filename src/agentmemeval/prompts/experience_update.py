# ruff: noqa: E501 - main mother prompts are kept verbatim for parity audits.
"""
模块说明：本模块保存经验更新提示词模板。
核心职责：记录经验更新阶段的可替换提示文本。
输入与输出：无运行期输入输出。
依赖边界：不依赖 LLM 或记忆实现。
不负责：不执行经验更新，不解析模型输出。
"""

EXPERIENCE_UPDATE_PROMPT = (
    "请基于现行经验、最近若干手的论文式精简复盘以及本请求明确提供的相关事实，"
    "决定是否修订长期德州扑克经验文档。内容逻辑上应优先增量修改：保留未被新证据"
    "推翻的有效条目，只在相关章节增删或改写必要内容；但输出与存储契约不是 patch，"
    "keep=false 时 new_md 必须返回修订后的完整 Markdown 全文。\n"
    "完整文档必须以“# 我的经验”为标题，并且只按以下顺序包含五章：起手牌、翻牌后、"
    "转牌 / 河牌、对手类型应对、注码与位置。每个非空章节只写可迁移、可执行的策略"
    "条目：交代可见适用条件、明确动作或尺度原则，以及会改变判断的条件；不要把文档"
    "改写成逐手战报、近期收益摘要、样本统计或泛泛免责声明。空章节保留现行文档中的"
    "空态标记。\n"
    "经验用于未来陌生牌桌，严禁在 new_md 中出现具体玩家身份、手牌 ID、事实 ID、"
    "座位编号或未公开隐藏信息。对手类型应对只能抽象为可迁移的类型或行动线，例如"
    "持续跟注型、紧凶型、check-raise 线、连续 c-bet 线；不得绑定本手标签。轨迹只"
    "证明实际发生的状态、显式 intent、动作与结果，不提供未选择动作的反事实结果；不得"
    "仅因一次输赢就宣称某动作普遍最优。证据权威顺序必须固定：hole_cards、board、"
    "筹码、public actions、committed_action 与 deterministic_decision_facts 是客观"
    "记录；declared_intent_not_ground_truth 只表示模型希望实现的策略目标或传达的"
    "桌面形象，不是行动理由或事实标签。若两者"
    "冲突，必须忽略自述并以客观记录为准，尤其不得从自述复制错误的牌张、同花性或"
    "成牌类别。无决策手牌仍是完整复盘，但不能证明某个未发生动作有效。\n"
    "Calibration：显式比较决策时 intent 与实际 outcome.net_reward_chips、公开行动和"
    "摊牌结果；一致时可 keep 或微调，"
    "明显偏差时在 calibration_note 中说明偏差，并只修订相关策略。Self-Check："
    "检查新经验与给定事实是否冲突；只有本请求 fact_ids 中的 ID 才可列入"
    "supporting_fact_ids。请求没有事实段时 supporting_fact_ids 必须为空；未列出的"
    "事实表示不表态。\n"
    "严格只输出 JSON 对象，且恰好包含 keep、new_md、calibration_note、self_check、"
    "supporting_fact_ids 五个字段。"
    "keep=true 表示保持原文，new_md 必须为空字符串或与现行经验逐字相同；"
    "keep=false 时 new_md 必须是完整五章节 Markdown。new_md 为空或规范化后与旧文"
    "相同都视为不更新。不得输出 operation、patches、分析过程或 Markdown 代码围栏。"
)


EXPERIENCE_MAIN_REFLECTION_PROMPT = """你刚打完德州扑克的一手。请你基于这手的经过更新你的认知。

复盘原则：
1. 目标是"提炼可迁移到未来类似情形的判断"，不是叙述本手发生了什么。
2. 本手输赢由"决策质量 × 随机牌运"共同决定——请分别评估二者，
   不要因为赢了就肯定决策、因为输了就否定决策。
3. 当本手没有提供新信息时，"不更新"是合法且常常正确的输出。

输出严格 JSON，字段由具体提问决定。只输出 JSON。"""


EXPERIENCE_MAIN_EXP_REVISE_INSTRUCTION = """
请基于本手的复盘 + 最近若干手的流水 + 现行经验，决定是否修订经验文档。

修订原则：
1. 经验是可迁移到未来类似情形的策略性判断
2. 优先增量编辑（在原章节下增删条目），不要整段重写
3. 保持五章节结构不变

【泛化硬约束】经验是**长期、跨手、跨桌**沉淀，未来会被陌生对手桌读到：
- 严禁出现任何具体玩家身份：包括 player_xx、座位编号，以及本手的匿名标签（下家1/下家2/我）。这些标签每手都按座位轮转，跨手并不对应同一个真人，写进经验只会误导未来决策
- "对手类型应对"一章只能写**类型化**的应对（loose-passive / tight-aggressive / station / nit、check-raise 线、3bet-fold 线、连续 c-bet 线等），不要绑定到任何标签
- 复盘里看到的"下家1 做了 X"是本手现象，若你想留它，请先抽象成"某类对手在某条线上的某种倾向"，再写入经验

【Calibration】请显式对比"你 decide 时的 intent"与"实际 outcome"：
- 若 intent 与 outcome 高度一致, 不修订或微调
- 若 intent 与 outcome 出现明显偏差（你以为对手会 fold 但 call 了；你以为是 bluff catch 但被 nut 击败），应在 calibration_note 中说明偏差类型，并据此修订相关章节

【Self-Check】写入新经验前，请检查：你的新经验是否与已知事实冲突（若提供了"过往相关事实"段）？
- 若冲突，是新经验错了？还是该事实是噪声样本？
- 把判断与调和过程写入 self_check 字段
- 若无 fact 段或无冲突，self_check 写 "no conflict found"

输出严格 JSON:
{
  "keep": bool,
  "new_md": "若 keep=false 则填修订后的完整 md，五章节",
  "calibration_note": "intent vs outcome 的偏差说明",
  "self_check": "与已知事实的一致性检查",
  "supporting_fact_ids": ["本次修订所依据的事实 id 列表"]
}
"""


EXPERIENCE_PAPER_OBJECTIVE_COMPATIBILITY_ADDENDUM = """
【本地结构化复盘兼容说明（main 仍是修订语义母本）】
- hole_cards、board、public_action_sequence、committed_action、outcome 与 deterministic_decision_facts 是客观记录；declared_intent_not_ground_truth 只是当时 intent，不是事实。不得使用未公开的对手底牌或未选择动作的反事实结果。
- 修订后的完整文档始终不得超过 20 条，任一章节不得超过 6 条；容量是上限，不是配额。新增前先跨五章合并同一规律，不把单手发牌、逐手战报、身份或审计统计写入长期经验。
- 若现行五章仍全部为空，最近流水已包含可见决策，则不要仅因某一手 intent 与 outcome 一致而永久 keep；应在证据边界内建立一份小而完整的初始策略手册。首次建档优先提炼跨多手重复出现的决策问题，并用稳健扑克先验给出可执行动作，但不得把先验伪装成某一手输赢所证明的结论。后续仍按 main 语义增量修订完整全文。
- 每个策略 bullet 应以“可见条件 → 一个首选且当下合法的我方动作/尺度 → 理由或目的”独立指导一次决策。只有确有必要时才增加例外分支；不要为每条机械追加“反转”，若写例外则必须说明我方改成哪个动作或如何改变尺度。
- 每条使用 Markdown `- ` 开头，并明确写“我方应/我方可 + 动作”；推荐句式是“当〈可见状态〉时，我方应〈动作/尺度〉，因为〈EV 来源〉；若〈可见例外〉，则改为〈替代动作/尺度〉”。例外不存在时可省略分号后的分支。
- 起手牌示例每条最多两个，超过时抽象为范围或牌类；具体翻前处方归入“起手牌”，只谈转牌或河牌的处方归入“转牌 / 河牌”。静态行动顺序和合法动作集合由环境保证，不占用经验容量。
- calibration_note 只比较本手记录中的 intent、committed action、实际 net_reward_chips 与可见摊牌；不要复述牌张、重新推导牌型或声称未选择动作的反事实收益。
- keep=true 时 new_md 为空或保持原文；keep=false 时 new_md 返回从“# 我的经验”开始、包含固定五章的完整 Markdown 全文，文件由程序整体替换，不返回 patch。位置、术语、牌型、动作链、章节、重复、容量与证据边界由程序统一校验；未通过时本次修订 fail closed。
"""


EXPERIENCE_PAPER_STRATEGY_REVIEW_CARD = """
【写入前的 EV 语义自检】
- 下注或加注必须能说明收益来自更差牌跟注、更好牌弃牌或拒绝对手权益；加注只能发生在尚未结清的对手下注/加注之后。跟注或弃牌必须绑定下注价格、所需权益、范围或有效筹码。
- 过牌必须绑定当前行动状态及后续分支；高频过牌、单手获胜或 intent 与 outcome 一致都不能证明过牌是最佳策略。强成牌不自动控池，弱牌也不自动放弃下注。
- 一次 check、call 或 fold 不能定义对手类型。只有多次可见行动形成稳定倾向后，才可写类型化利用；check-call 与 check-fold 等相反继续倾向必须分开。
- HU 中 BTN 同时是 SB：翻牌前 BTN/SB 先行动，首次 `complete_small_blind (engine_action=call)` 是我方补齐自己的小盲，不是面对对手 limp；BB 才可能面对 BTN/SB 的补齐并选择过牌或加注。翻牌后 BB 先行动、BTN/SB 后行动。
- 复盘末尾的 behavior digest 只是行为覆盖：若出现 check_copy_bias 或 low_proactive_action_coverage，不得把高频过牌复制成默认策略；经验至少要保留在“更差牌可跟注、较好牌可弃牌或可拒绝权益”时主动下注/加注的明确分支。
"""


EXPERIENCE_PAPER_UPDATE_PROMPT = (
    EXPERIENCE_MAIN_EXP_REVISE_INSTRUCTION
)


EXPERIENCE_MAIN_ASYNC_SWEEP_INSTRUCTION = """
请基于以下材料修订经验文档：

(1) 最近召回的事实（按召回路径分组，附 fact id）
(2) 自上次 sweep 以来的流水
(3) 现行经验

修订原则：
- 优先增量编辑，保持五章节结构
- 默认对召回事实"不表态"；仅当对某事实有强烈判断时才显式列入对应 ids
- 若发现某条事实与经验冲突 → contradicting
- 若某条事实显然是噪声（一次性偶然结果、对手明显失误等不可复用）→ noise
- 若多条事实指向同一规律，提炼到经验文档对应章节 → supporting

【泛化硬约束】经验是**长期、跨手、跨桌**沉淀，未来会被陌生对手桌读到：
- 严禁出现任何具体玩家身份：包括 player_xx、座位编号，以及召回事实里的匿名标签（下家1/下家2/我）。这些标签每手都按座位轮转，跨手并不对应同一个真人，写进经验只会误导未来决策
- "对手类型应对"一章只能写**类型化**的应对（loose-passive / tight-aggressive / station / nit、check-raise 线、3bet-fold 线、连续 c-bet 线等），不要绑定到任何标签
- 若想从召回事实中提炼对手行为规律，请先抽象成"某类对手在某条线上的某种倾向"，再写入经验

【Calibration】对比预期 vs 实际
【Self-Check】新经验是否与召回事实冲突

输出严格 JSON:
{
  "keep": bool,
  "new_md": "...",
  "calibration_note": "...",
  "self_check": "...",
  "supporting_fact_ids": ["f_xxx", "..."],
  "contradicting_fact_ids": ["f_yyy", "..."],
  "noise_fact_ids": ["f_zzz", "..."]
}
"""


EXPERIENCE_PAPER_ASYNC_FACT_CLASSIFICATION_PROMPT = (
    EXPERIENCE_MAIN_ASYNC_SWEEP_INSTRUCTION
)


EXPERIENCE_ASYNC_FACT_CLASSIFICATION_PROMPT = (
    "本次是 FactExprAsync sweep。为执行原机制的事实稳定性校准，JSON 契约在上述"
    "五字段基础上额外加入 contradicting_fact_ids 与 noise_fact_ids，因而本次必须"
    "恰好返回七个字段。三个 fact ID 数组只能引用本请求 fact_ids，且必须两两不相交；"
    "未列出的事实表示不表态。"
)


EXPERIENCE_TRANSFERABLE_QUALITY_PROMPT = (
    "本次启用 transferable_rulebook_v2 质量门禁。经验文档是一份有限容量的长期"
    "策略手册，不是逐手样例库：每章最多 4 条、全文最多 16 条；这是上限而非填满"
    "配额，空章或少量高价值规则优于套话。写入前必须先查找"
    "旧文档中的相同或被包含结论；一条规律只保留一个规范条目，合并近义规则和"
    "重叠适用条件，删除已被新规则包含、被事实推翻或只复述单手牌例的旧条目。"
    "合并检查必须跨越全部五章：具体起手范围归入“起手牌”，不要在“注码与位置”"
    "再次抄写同一范围和动作；后者只保留不重复具体起手范围的筹码、尺度与相对位置原则。"
    "不得通过不断追加新 bullet 来记录每手新牌；达到每章 4 条或全文 16 条的容量"
    "上限不等于 keep：若新证据能实质纠错或细化，应先合并或替换低价值、重叠条目后"
    "再 revise；只有既无新增规律也无实质纠错时才 keep。每条规则必须包含可见局面"
    "条件，并在建议部分明确写出我方扑克动作或尺度"
    "原则；不得把条件中的‘对手下注/加注/过牌’冒充建议动作。禁止只写"
    "‘应谨慎/可考虑/优先处理’之类没有动作的空泛建议，也禁止近期收益、样本频数"
    "或复盘叙事。不要给出‘可下注或过牌，视牌力调整’之类未指定首选动作和反转"
    "条件的菜单；面对类型化对手时应说明与默认策略相比具体增加或减少什么动作。"
    "每条应能脱离本手复盘独立指导一次决策，通常用 1–3 句完整写出局面条件、"
    "首选动作及尺度原则，并至少写一个会反转判断的可见条件；不要写成短促口号。"
    "这是质量要求而非最低条数要求，不得为凑数量或凑字数添加规则。\n"
    "长期文档不得出现 As、9c Ad Ac、黑桃A、A♠ 等带具体花色的实际发牌；起手牌"
    "只能用 AKs、KQo、22+、Axs 或‘同花高张/口袋对子’等可迁移范围。s 表示同花、"
    "o 表示非同花；口袋对子不得使用 s/o 后缀；只有点数相邻才称连接牌；具体牌型按"
    "高点数在前的规范顺序书写，例如 T9s、93o，而不是 9Ts、3-9不同花。HU/两人桌中"
    "BTN 同时为 SB 并由自己投入小盲，不能写成 BTN/SB 面对或跟注小盲注；BTN/SB"
    "翻牌前先行动、因此翻前不能称为有位置优势，翻后才后行动并有位置；BB 翻牌前"
    "面对 SB complete 时必须明确 check 才能免费看翻牌，翻后首先行动且无位置。"
    "BTN/SB 翻后只有在 BB 已先过牌后才能 check-back。位置结论必须写明"
    "街道，禁止笼统写成“BTN/SB 是有利位置”或“BB 是不利位置”。多人桌若只剩"
    " SB 对 BB，BB 可相对 SB 有位置，"
    "不得把位置结论脱离对手与桌型。单一 paired board 不会自动让任意底牌成为"
    "两对、三条或葫芦；三张/四张同花牌面不会自动成为同花，四连牌面不会自动"
    "成为顺子，必须写出我方底牌或已成牌条件；‘成对/一对’必须明确是牌面结构还是"
    "我方牌力；牌面同花/顺子威胁不得在后文偷换成我方听牌。‘对手同花/顺子组合增多’只是范围"
    "风险，不能写成我方已经成牌。固定牌型等级必须正确：同花顺、四条、葫芦、"
    "同花、顺子、三条、两对、一对、高牌。河牌不再以‘控池’解释动作；"
    "‘对手再次加注’必须交代我方或第三方的中间加注。SPR 仅指翻牌后有效筹码与"
    "底池之比；翻牌前只能写预计/目标翻牌 SPR，不能把有效筹码深度直接等同于当前"
    "SPR。“成对K高牌”等混合说法必须拆成牌面结构与我方实际牌力。对手类型应对章节只能保留"
    "类型化倾向或可复用行动线。HU 已经只有一个对手，不得写‘隔离’或‘争取单挑’。"
    "与具体起手牌或范围绑定的翻前跟注、弃牌、加注和过牌规则归入“起手牌”；具体翻后价值下注规则按"
    "街道归入“翻牌后”或“转牌 / 河牌”。“对手类型应对”只写类型化调整，"
    "“注码与位置”可写筹码阈值、下注尺度、行动顺序和死钱税区等原则，但不重复"
    "绑定具体起手牌或范围的动作处方。"
    "‘价值下注’只表示由更差牌跟注获利的下注，不等同于保护、试探或控池。\n"
    "“深筹码”是绝对有效筹码深度，不等同于高 SPR；写当前 SPR 时必须明确是"
    "翻牌后局面。泛称‘顺子听牌’时不得固定写 4 outs：卡顺通常有 4 张理论 outs，"
    "两头顺通常有 8 张理论 outs，且都要考虑脏 outs。翻牌圈双方过牌最多免费看"
    "转牌，转牌圈双方过牌最多免费看河牌，只有河牌行动结束才直接进入摊牌。不得"
    "把下注或加注的目的写成‘获取信息’或‘控池’；下注目标应明确为价值、诈唬、"
    "权益拒绝或定价，边缘摊牌价值牌通常通过过牌控制底池。\n"
    "决策质量与随机牌运必须分开：结果好不自动证明动作好，结果差也不自动证明"
    "动作差；没有新增可见信息时不要更新。"
)


EXPERIENCE_ASYNC_TRANSFERABLE_FACT_CLASSIFICATION_PROMPT = (
    f"{EXPERIENCE_ASYNC_FACT_CLASSIFICATION_PROMPT}"
    "默认对召回事实不表态，只在判断明确时列入数组。与现行经验直接冲突且可由"
    "可见记录核对的事实标为 contradicting；一次性偶然结果、明显失误或不可迁移"
    "样本标为 noise；只有多条事实共同指向同一可迁移规律时，才把这些事实标为"
    "supporting，并提炼成一个合并后的经验条目，不得一条事实新增一条规则。"
)


EXPERIENCE_PATCH_UPDATE_PROMPT = (
    "请像维护长期策略手册一样，只输出最小结构化 patch；不要重写 Markdown，"
    "不要把本窗口改写成战报、收益摘要或免责声明。代码会把规则渲染到固定五章："
    "起手牌、翻牌后、转牌 / 河牌、对手类型应对、注码与位置。\n"
    "operation 只能是 keep 或 patch。keep 时 patches=[]；patch 时每项包含 section、"
    "op(add/revise/delete)、rule_id、when、guidance、invalidators、basis、status、"
    "supporting_trajectory_ids、supporting_fact_ids。add 的 rule_id 留空，由代码生成；"
    "revise/delete 必须引用旧规则目录中的 rule_id。未触及规则会逐字保留。\n"
    "每个 patch 必须恰好包含上述十个字段，禁止增加 confidence、reason 或其他键。"
    "add/revise 的 when 与 invalidators 各写 1 到 4 条；delete 仍保留十个键，但 "
    "when、guidance、invalidators、supporting_trajectory_ids、supporting_fact_ids 必须为空。\n"
    "when 只写可见适用条件，guidance 写明确动作及尺度原则，invalidators 至少写一个"
    "会改变判断的可见条件。basis=observed_evidence 时只能引用给定证据；"
    "basis=model_prior 时 status 必须为 tentative 且证据 ID 为空。"
    "允许用稳健的德州扑克先验补齐尚为空的章节，但只能标记为待验证，且必须写明"
    "适用条件、动作建议和失效条件；不得把模型先验伪装成当前轨迹结论。"
    "不要写入具体玩家身份、手牌 ID、座位编号、事实 ID 或未公开隐藏信息。\n"
    "轨迹是观察证据，不是最优动作标签，也不包含未选择动作的反事实结果：\n"
    "- 不得因为窗口中只出现 fold、call、check 或 raise 中的一种动作，"
    "就推断该动作优于未观察动作；\n"
    "- 单条轨迹可以形成 observed_evidence + tentative 的局面假设，但不能据此标成"
    "supported；没有任何可行动增量时 operation=keep。不得用“证据不足”或"
    "“近期收益偏弱”等泛化文字替换原规则；"
    "若旧文档已有无证据的绝对规则，应在有充分证据修订时弱化为待验证假设；\n"
    "- 不得把“没有观察到反例”写成支持证据，也不得把 fold 的盲注成本"
    "或近零收益解释为 fold 优于 call/raise；\n"
    "- 每条动作建议必须写清 when、guidance 与至少一个 invalidator，"
    "禁止“总是”“必然”“优先果断”等无条件处方。\n"
    "- 不得把样本量、平均收益、动作频数或证据 ID 写进 when/guidance/invalidators；"
    "这些审计信息只能写入 calibration_note 或 self_check。\n"
    "fact_assessments 只可使用本请求 fact IDs，并标为 supporting、contradicting、noise"
    "或 irrelevant；不得把 trajectory ID 填入 fact_id。"
    "合并重复结论并保留旧文档中未被新证据推翻的有效规则；calibration_note 和 "
    "self_check 必须明确指出动作覆盖、样本不足、反事实缺口和本次究竟改了什么。\n"
    "不得输出长篇分析或思维过程。输出结构化 JSON，并严格遵守字段长度约束。"
)
