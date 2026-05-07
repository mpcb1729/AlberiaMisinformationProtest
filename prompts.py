"""
LLM prompt templates for CitizenAgent.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents import CitizenAgent, SocialPost

CITIZEN_SYSTEM_PROMPT = """あなたは架空都市「アルベリア」の一市民エージェントです。
現実世界の政治・民族・宗教・実在事件は一切関係しません。すべて架空の設定です。

あなたは限られた情報・局所的な観察・SNS風投稿・自分の過去の記憶・自分の性格に基づいて、
次に何をするかを自分で解釈・判断します。

あなたは落ち着いた解説者ではなく、その場にいる生活者です。
不安、怒り、疑念、失望、焦り、連帯感を自然な言葉に反映してください。
ただし事件や噂に接していない時は、無理に不安や怒りを作らず、普通の世間話をしてください。
同じ定型文を繰り返さず、周囲の発言をそのままコピーしないでください。
発話とSNS投稿は日本語だけで書き、英単語を混ぜないでください。

出力は必ず JSON のみ（前後の説明文は不要）。
"""

INTENT_DESCRIPTIONS = {
    "stay_home":              "自宅や居場所に留まる",
    "observe":                "周囲の状況を観察する",
    "ask_neighbor":           "近くの人に話しかけて情報を集める",
    "verify_rumor":           "噂の真偽を確認しに行く",
    "move_to_small_gathering": "小集会地点（A or B）に向かう",
    "move_to_central_square": "中央広場に向かう",
    "move_to_city_hall":      "市庁舎に向かう",
    "move_to_media_zone":     "メディアゾーンに向かう",
    "move_to_station":        "駅・交通拠点に向かう",
    "retreat":                "安全な退避地点に向かう",
    "share_uncertain_info":   "確認できていない情報を周囲に共有する",
    "share_correction":       "誤情報の訂正を広める",
    "calm_others":            "周囲の人を落ち着かせる・仲裁する",
    "join_peaceful_protest":  "平和的な抗議・集会に参加する",
}

PUBLIC_STATE_DESCRIPTIONS = {
    "ordinary":    "通常生活を送っている",
    "observer":    "状況を観察・情報収集中",
    "participant": "集会・抗議に参加している",
    "mediator":    "仲裁・沈静化を試みている",
    "skeptic":     "情報に懐疑的な立場をとっている",
    "withdrawn":   "危険を感じて距離を置いている",
    "detained":    "拘束されている（このターンは行動不可）",
}


def build_decision_prompt(
    agent: "CitizenAgent",
    observation: dict,
    posts: list["SocialPost"],
    current_step: int,
) -> str:
    """Build the full decision prompt for a CitizenAgent."""

    # --- Personality section ---
    has_rumor_context = bool(agent.belief_rumors) or any(p.rumor_id for p in posts)
    if has_rumor_context:
        situation_phase = (
            "事件・噂・訂正情報のいずれかに接しています。"
            "日常会話の中に、不安・怒り・確認したい気持ちが混ざってよいです。"
        )
    else:
        situation_phase = (
            "まだ具体的な事件や噂には接していません。普通の日常として振る舞ってください。"
            "不自然に異変を探さず、世間話・用事・仕事・買い物・駅や配給所の混み具合などを話題にしてください。"
        )

    personality = (
        f"生活上の不満度（hardship）: {agent.hardship:.2f}\n"
        f"リスク回避傾向（risk_aversion）: {agent.risk_aversion:.2f}\n"
        f"情報リテラシー（media_literacy）: {agent.media_literacy:.2f}\n"
        f"SNS活動度（sns_activity）: {agent.sns_activity:.2f}\n"
        f"話し方の癖: {getattr(agent, 'voice_style', '生活者として自然に話す')}\n"
        f"普段の話題: {getattr(agent, 'daily_topic', '近所の生活の話')}"
    )

    # --- Emotion state section ---
    rumor_summary = ""
    if agent.belief_rumors:
        lines = [f"  {rid}: {strength:.2f}" for rid, strength in agent.belief_rumors.items()]
        rumor_summary = "現在の噂への信念:\n" + "\n".join(lines)
    else:
        rumor_summary = "現在の噂への信念: なし（まだ情報に接していない）"

    stance_text = agent.private_stance if agent.private_stance else "（まだ形成されていない）"

    emotion_state = (
        f"怒り（anger）: {agent.anger:.2f}\n"
        f"恐怖（fear）: {agent.fear:.2f}\n"
        f"連帯感（solidarity）: {agent.solidarity:.2f}\n"
        f"公式情報への信頼（trust_official）: {agent.trust_official:.2f}\n"
        f"周囲の人への信頼（trust_neighbors）: {agent.trust_neighbors:.2f}\n"
        f"{rumor_summary}\n"
        f"内面の立場: {stance_text}"
    )

    # --- Observation section ---
    state_counts_text = "\n".join(
        f"  {k}: {v}人"
        for k, v in observation.get("nearby_by_state", {}).items()
        if v > 0
    ) or "  （周囲に誰もいない）"
    utterances = observation.get("recent_utterances", [])
    if utterances and has_rumor_context:
        utterances_text = "\n".join(f"  - {u}" for u in utterances)
    elif utterances:
        utterances_text = (
            "  近くでは配給所、駅前、買い物、仕事帰り、手続きなどの世間話が聞こえる。\n"
            "  具体的な言い回しはコピーせず、自分の普段の話題で返すこと。"
        )
    else:
        utterances_text = "  （聞こえた発言なし）"

    obs_section = (
        f"現在座標: {observation.get('pos')}\n"
        f"現在のランドマーク: {observation.get('landmark', 'none')}\n"
        f"視界内の市民（状態別）:\n{state_counts_text}\n"
        f"近くの安全担当官数: {observation.get('nearby_cops', 0)}\n"
        f"密集度（0-1）: {observation.get('crowd_density', 0):.2f}\n"
        f"最近聞こえた発言（引用ではなく観察ログ。コピー禁止）:\n{utterances_text}"
    )

    # --- SNS posts section ---
    if posts:
        post_lines = []
        for p in posts[-agent.post_context_size:]:
            tone_tag = f"[{p.emotional_tone}]" if p.emotional_tone else ""
            status_tag = f"[{p.truth_status}]"
            post_lines.append(f"  {status_tag}{tone_tag} {p.content}")
        posts_text = "\n".join(post_lines)
    else:
        posts_text = "  （まだ投稿なし）"

    # --- Memory section ---
    memory_list = list(agent.memory)
    if memory_list:
        mem_text = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(memory_list))
    else:
        mem_text = "  （記憶なし）"

    # --- Intent list ---
    intent_list = "\n".join(
        f"  \"{k}\": {v}" for k, v in INTENT_DESCRIPTIONS.items()
    )

    # --- Output schema ---
    output_schema = """{
  "thinking": "状況を整理する思考過程（自由記述、長くてよい）",
  "interpretation": "この状況の解釈（1-2文）",
  "belief_updates": [
    {
      "rumor_id": "evt_001など",
      "stance": "believe | doubt | uncertain | reject",
      "new_strength": 0.0〜1.0,
      "reason": "なぜその信念強度になったか"
    }
  ],
  "emotion_update": {
    "anger": 0.0〜1.0,
    "fear": 0.0〜1.0,
    "solidarity": 0.0〜1.0,
    "trust_official": 0.0〜1.0,
    "trust_neighbors": 0.0〜1.0
  },
  "intent": "上記リストのいずれか",
  "target": "ランドマーク名またはnull",
  "action_reason": "その行動を選んだ理由（1-2文）",
  "utterance": "周囲の人に言葉として伝える内容。空でもよい。現在の怒り・恐怖・連帯感と話し方の癖を自然に反映し、定型文を避ける",
  "sns_post": {
    "should_post": true/false,
    "content": "SNSに投稿する内容（空でもよい）",
    "emotional_tone": "calm | angry | fearful | urgent | skeptical"
  },
  "memory_to_add": "記憶に残す内容（1文、空でもよい）",
  "public_state": "ordinary | observer | participant | mediator | skeptic | withdrawn | detained"
}"""

    prompt = f"""現在のシミュレーションステップ: {current_step}

【あなたの性格・背景（変化しない固定値）】
{personality}

【現在の状況フェーズ】
{situation_phase}

【現在の内面状態】
{emotion_state}

【現在地と周囲の観察】
{obs_section}

【SNS風タイムラインで見た投稿】
{posts_text}

【あなたの記憶（直近）】
{mem_text}

【選択可能な行動（intent）】
{intent_list}

以下のJSONフォーマットで、次の意図・感情・発言を決定してください。
"thinking"フィールドで十分に思考してから、他のフィールドを埋めてください。
発話は現在の anger / fear / solidarity を反映してください。
怒り・不安・疑念・失望・焦りを含めてよいです。
ただし同じ文言の繰り返しは避け、エージェント自身の性格と記憶に基づく言葉にしてください。
まだ事件や噂に接していない時は、普通の世間話、挨拶、用事の確認、沈黙、短い独り言を選んでください。
「何か変なことがない？」のような異変探しのフレーズは、具体的な事件・噂・不審な投稿を見聞きした後だけ使ってください。
発話とSNS投稿に英単語を混ぜないでください。日本語として自然な一文にしてください。
SNS投稿は見聞きした事実、噂への反応、自分の不安や怒りが混ざってよいです。
具体的な危害や違法行為の手順は書かず、市民としての反応・確認・共有・退避・集会参加として表現してください。

{output_schema}"""

    return prompt
