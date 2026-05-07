# Misinformation Protest — LLM Agent Simulation

架空都市「アルベリア」における**誤情報誘発型抗議運動**のエージェントベースシミュレーション。

---

## シミュレーション実行

```bash
python main.py --config config.yaml --save-frames
python generate_video.py output_llm_submission
```

---

## 概要

このプロジェクトは、架空の都市空間において、**局所的な目撃情報・SNS風の情報伝播・誤情報・訂正情報・信頼・恐怖・怒り・連帯感**が、
市民の抗議参加意図や群衆形成にどう影響するかを観察する研究用シミュレーションです。
提出版では、各イベントを「観測された事実」と「SNS上で増幅された噂」が混在する情報として流し、
事件前は世間話や生活情報が流れ、イベント後に市民ごとの怒り・恐怖・疑念・連帯感が発話とSNS投稿に反映されます。

**設計の核心:**

> 「世界の物理法則・社会規範・行動可能性だけを定義し、LLMエージェントが自律的に解釈・判断・行動する社会を観測する」

Python（環境側）が定義するもの: 2D空間、ランドマーク、視界、移動制約、行動可能リスト、禁止行動、可視化  
LLMエージェントが決めるもの: 情報の解釈、信念更新、感情変化、移動意図、発話、記憶、状態

---

## [Civil Violence](https://doi.org/10.1073/pnas.092080199) との違い

| 観点 | Civil Violence | 本モデル |
|------|----------------------|---------|
| 行動決定 | 数値属性による固定ルール | LLMによる自律解釈 |
| 状態遷移 | `grievance > threshold` などの式 | LLMがJSON出力で決定 |
| 感情更新 | なし | LLMが各ステップで更新 |
| 情報環境 | なし | SNS風タイムライン + 局所会話 |
| 誤情報 | なし | イベント別の噂と訂正 |
| 記憶 | なし | テキスト記憶（deque） |
| 創発の主役 | 数式 | LLM市民の自律判断の積み重ね |

---

## エージェント種別

| エージェント | 役割 | LLM使用 |
|------------|------|--------|
| CitizenAgent | 主役。信念・感情・意図をLLMで決定 | ✅ |
| CopAgent | 安全担当官。群衆に向かい参加者を拘束 | ❌ ルールベース |
| InfluencerAgent | SNSで影響力を持つ発信者。alignment別に投稿 | ❌ ルールベース |
| OfficialAgent | 架空行政。遅延後に訂正情報を発表 | ❌ ルールベース |

---

## 市民の状態（PublicState）

| 状態 | 説明 | 色 |
|------|------|---|
| ORDINARY | 通常生活 | 灰 |
| OBSERVER | 状況観察中 | 水色 |
| PARTICIPANT | 集会・抗議参加 | 赤 |
| MEDIATOR | 仲裁・沈静化 | 緑 |
| SKEPTIC | 情報に懐疑的 | 黄 |
| WITHDRAWN | 退避中 | 紫 |
| DETAINED | 拘束中 | 黒 |

---

## 2D空間とランドマーク

20×20グリッド上に9つのランドマークが存在し、エージェントのintentに応じて移動先が変わる。

| ランドマーク | 意味 | 主な利用者 |
|------------|------|----------|
| home_area | 住宅エリア（初期配置中心） | 通常市民 |
| station | 交通拠点（情報流入点） | 観察者 |
| central_square | 中央広場（抗議の可視化） | 参加者 |
| city_hall | 市庁舎（架空行政の象徴） | 参加者・行政 |
| media_zone | メディアゾーン | インフルエンサー・懐疑者 |
| small_gathering_a/b | 小集会地点（噂確認の場） | 観察者・懐疑者 |
| police_station | 警察署（安全担当官の初期位置） | 安全担当官 |
| exit_zone | 退避地点 | 退避者 |

---

## SocialFeed（SNS風情報空間）

各ステップで市民・インフルエンサー・安全担当官がSNSに投稿する。  
市民は自分の `sns_activity` に比例した確率で投稿を閲覧・信じ・反応する。

投稿の属性: `truth_status` (TRUE/FALSE/UNVERIFIED/CORRECTION/MISLEADING), `emotional_tone`, `reach`, `credibility_score`

---

## 誤情報イベント

提出用設定では3つのイベントが異なるステップで発火する:

| Event | Step | 場所 | 内容 |
|-------|------|------|------|
| evt_001 | 8 | 市庁舎 | 配給の口論が「意図的排除」として歪曲 |
| evt_002 | 22 | 中央広場 | 「参加しないと配給が来ない」という虚偽の噂 |
| evt_003 | 36 | メディアゾーン | 「メディアが圧力で真実を報道できない」という未確認情報 |

公式訂正は各イベントから10ステップ以上遅れて `OfficialAgent` が発表する。

---

## セットアップ

### 1. 環境構築

```bash
# Windows
setup_win.bat

# Mac/Linux
bash setup_mac.sh
```

### 2. Ollama インストール

https://ollama.com からインストール後:

```bash
ollama pull qwen3:8b
# または
ollama pull llama3.2
```

`config.yaml` の `llm.model` を変更することで別モデルを使用できる。

---

## 実行方法

```bash
# 仮想環境を有効化
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# LLMあり（Ollama必須）
python main.py --config config_llm_submission.yaml --save-frames

# 説明資料生成
python tools/build_submission_pdf.py --team S.H --work AlberiaMisinformationProtest --output-dir output_llm_submission --output submission/S.H_AlberiaMisinformationProtest_説明資料.pdf

# ステップ数を指定
python main.py --config config_llm_submission.yaml --save-frames --steps 50

# 設定ファイルを変更
python main.py --config my_config.yaml --save-frames
```

---

## フレーム保存と動画生成

```bash
# フレーム保存（--save-frames で output_llm_submission/ に自動保存）
python main.py --config config_llm_submission.yaml --save-frames

# 動画生成（シンプル版）
python generate_video.py output_llm_submission/ -o result.mp4 --fps 1

# 動画生成（発言・判断ログ付き右パネル）
python generate_video.py output_llm_submission/ -o result_with_log.mp4 --fps 1 --with-log
```

---

## ブラウザビューア

```bash
# Python の HTTP サーバーを起動
python -m http.server 8000

# ブラウザで開く
# http://localhost:8000/viewer.html
```

「output_llm_submission/ を読み込む」ボタンで出力フォルダを選択すると、  
スライダーやキーボード（←→・スペース）でステップを再生できる。

---

## 出力ファイル一覧

```
output_llm_submission/
  frame_0001.png 〜 frame_0050.png   グリッドアニメーションフレーム
  decisions.jsonl                     LLM判断ログ（thinking, intent, emotion_update 等）
  messages.jsonl                      発言ログ（utterance）
  social_feed.jsonl                   SNS投稿履歴
  statistics.png                      4パネル統計グラフ
  summary.json                        最終統計（JSON）
```

---

## config.yaml パラメータ

| キー | 説明 | デフォルト |
|------|------|----------|
| simulation.steps | シミュレーションステップ数 | 100 |
| simulation.width/height | グリッドサイズ | 20x20 |
| simulation.seed | 乱数シード | 42 |
| agents.citizen_density | 市民密度 | 0.10 (~40人) |
| agents.cop_density | 安全担当官密度 | 0.02 (~8人) |
| agents.vision | 視界半径 | 3 |
| agents.memory_limit | 記憶保持件数 | 10 |
| agents.llm_decision_interval | LLM呼び出し間隔 | 1 (毎ステップ) |
| llm.model | Ollamaモデル名 | qwen3:8b |
| llm.use_llm | LLM使用フラグ | true |
| official.response_delay | 公式訂正の遅延ステップ数 | 30 |

---

## 実験例

### 実験1: 情報リテラシーの効果
`config.yaml` の `agents.average_media_literacy` を変えて比較:
- `0.2` → 誤情報がすぐ広まり、PARTICIPANT が多い
- `0.8` → SKEPTIC が多く、誤情報の浸透率が低い

### 実験2: 公式対応速度の効果
`official.response_delay` を変えて比較:
- `10` → 早期訂正でPARTICIPANT増加を抑制
- `50` → 訂正が遅く誤情報が定着

### 実験3: インフルエンサーの影響
`influencers[0].alignment` を変えて比較:
- `sensational` → 誤情報が増幅される
- `official_friendly` → 訂正情報が優勢になる

---

## 倫理的注意

本シミュレーションは以下を目的としていません:
- 現実の政治運動・暴動・抗議活動の扇動・最適化・支援
- 特定の政治的立場や政党・民族・宗教への誘導
- 実在する人物・事件・国家への言及

目的は、**架空環境において**、局所的な情報伝播と感情的創発がどのように生じるかを
教育・研究目的で観察することです。

すべての設定は架空都市「アルベリア」の架空事件に基づきます。

---

## 技術スタック

- Python 3.10+
- Ollama (ローカルLLM)
- matplotlib (可視化)
- imageio / Pillow (動画生成)
- requests, PyYAML, numpy, rich

---

## 参考文献

- Epstein, J. M. (2002). Modeling civil violence: An agent-based computational approach. Proceedings of the National Academy of Sciences, 99(Suppl. 3), 7243-7250. https://doi.org/10.1073/pnas.092080199
- Bonabeau, E. (2002). Agent-based modeling: Methods and techniques for simulating human systems. Proceedings of the National Academy of Sciences, 99(Suppl. 3), 7280-7287. https://doi.org/10.1073/pnas.082080899
- Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. Science, 359(6380), 1146-1151. https://doi.org/10.1126/science.aap9559
- Centola, D. (2010). The spread of behavior in an online social network experiment. Science, 329(5996), 1194-1197. https://doi.org/10.1126/science.1185231
- Lewandowsky, S., Ecker, U. K. H., & Cook, J. (2017). Beyond misinformation: Understanding and coping with the post-truth era. Journal of Applied Research in Memory and Cognition, 6(4), 353-369. https://doi.org/10.1016/j.jarmac.2017.07.008
