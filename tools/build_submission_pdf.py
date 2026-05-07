from __future__ import annotations

import argparse
import json
import os
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_font() -> FontProperties:
    candidates = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return FontProperties(fname=candidate)
    return FontProperties()


JP_FONT = load_font()


def wrapped_lines(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in str(text).splitlines():
        if not raw_line:
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                raw_line,
                width=width,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
    return lines


def add_text(
    ax,
    x,
    y,
    text,
    size=18,
    weight="normal",
    color="#111827",
    ha="left",
    width=42,
    line_height=None,
):
    line_height = line_height or size * 0.0045
    for line in wrapped_lines(text, width):
        ax.text(
            x,
            y,
            line,
            fontsize=size,
            fontproperties=JP_FONT,
            fontweight=weight,
            color=color,
            ha=ha,
            va="top",
        )
        y -= line_height
    return y


def setup_slide(title: str):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="#f8fafc")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0.91), 1, 0.09, color="#111827"))
    add_text(ax, 0.045, 0.965, title, size=20, weight="bold", color="white", width=52)
    return fig, ax


def bullet_list(ax, items, x=0.07, y=0.78, gap=0.025, size=17, width=46):
    for item in items:
        y = add_text(ax, x, y, f"・{item}", size=size, width=width)
        y -= gap
    return y


def read_summary(output_dir: Path) -> dict:
    path = output_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_decision_sources(output_dir: Path) -> Counter:
    counts: Counter = Counter()
    path = output_dir / "decisions.jsonl"
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        counts[rec.get("decision_source", "unknown")] += 1
    return counts


def add_architecture(ax):
    boxes = [
        ("config_llm_submission.yaml\n設定・イベント", 0.08, 0.63),
        ("Simulation\nグリッド・イベント・ログ", 0.33, 0.63),
        ("CitizenAgent\nLLM判断・記憶・発話", 0.58, 0.63),
        ("Ollama qwen3:8b\nJSON意思決定", 0.58, 0.36),
        ("SocialFeed\nSNS風投稿・訂正情報", 0.33, 0.36),
        ("Visualization\nPNGフレーム・MP4", 0.08, 0.36),
    ]
    for label, x, y in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                0.19,
                0.13,
                facecolor="white",
                edgecolor="#334155",
                linewidth=1.6,
            )
        )
        add_text(ax, x + 0.015, y + 0.105, label, size=13, width=16)
    arrows = [
        ((0.27, 0.695), (0.33, 0.695)),
        ((0.52, 0.695), (0.58, 0.695)),
        ((0.675, 0.63), (0.675, 0.49)),
        ((0.58, 0.425), (0.52, 0.425)),
        ((0.33, 0.425), (0.27, 0.425)),
        ((0.42, 0.63), (0.42, 0.49)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", color="#0f766e", lw=2),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="S.H")
    parser.add_argument("--work", default="AlberiaMisinformationProtest")
    parser.add_argument("--output-dir", default="output_llm_demo")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    submission_dir = PROJECT_ROOT / "submission"
    submission_dir.mkdir(exist_ok=True)
    pdf_path = Path(args.output) if args.output else submission_dir / f"{args.team}_{args.work}_説明資料.pdf"

    summary = read_summary(output_dir)
    source_counts = count_decision_sources(output_dir)

    with PdfPages(pdf_path) as pdf:
        fig, ax = setup_slide(f"{args.team} / {args.work}")
        add_text(ax, 0.07, 0.76, "誤情報と市民行動のLLMシミュレーション", size=23, weight="bold", width=46)
        add_text(ax, 0.07, 0.52, "架空都市アルベリアで、噂・訂正・感情の変化を観察する", size=18, color="#334155", width=54)
        add_text(ax, 0.07, 0.35, f"チーム名: {args.team}\n作品名: {args.work}", size=19, width=42)
        add_text(ax, 0.07, 0.17, "提出デモは Ollama / qwen3:8b によるLLM実行で生成", size=16, color="#0f766e", width=54)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = setup_slide("コンセプト")
        bullet_list(
            ax,
            [
                "事件前は、買い物・駅前・手続きなどの日常会話が流れる",
                "イベント後は、観測された事実とSNS上の噂が同時に広がる",
                "市民は、観察・記憶・SNS投稿をもとにLLMで次の行動を判断する",
                "発話には、怒り・恐怖・疑念・連帯感が反映される",
                "目的は、架空環境で情報伝播と集団行動の構造を可視化すること",
            ],
            size=15,
            width=58,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = setup_slide("システム構成")
        add_architecture(ax)
        add_text(ax, 0.08, 0.19, "JSONLログに、各市民の解釈・感情更新・意図・発話・判断ソースを保存", size=16, width=58)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = setup_slide("エージェント")
        bullet_list(
            ax,
            [
                "CitizenAgent: 感情・記憶・噂への信念を持つLLM市民",
                "InfluencerAgent: 立場に応じてSNS投稿を行う発信者",
                "OfficialAgent: 一定遅延後に訂正情報を投稿する公式役",
                "CopAgent: 群衆へ移動するルールベース安全担当官",
            ],
            size=15,
            width=58,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = setup_slide("提出デモ設定")
        bullet_list(
            ax,
            [
                "config_llm_submission.yaml を使用",
                "市民10人、50ステップ、誤情報イベント3件",
                "市民判断は毎ステップ Ollama / qwen3:8b に委譲",
                "各判断は JSON として保存し、動画右側にログ表示",
                "出力はフレーム、SNS投稿、発話、判断ログ、統計として保存",
            ],
            size=15,
            width=58,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = setup_slide("実行結果")
        if summary:
            result_items = [
                f"総ステップ数: {summary.get('total_steps_run', '-')}",
                f"最大抗議参加者数: {summary.get('peak_participant_count', '-')}人（Step {summary.get('peak_participant_step', '-')}）",
                f"最終状態: 観察中 {summary.get('observer_count', '-')}人、抗議参加 {summary.get('participant_count', '-')}人、拘束 {summary.get('detained_count', '-')}人",
                f"平均感情: 怒り {summary.get('average_anger', '-')} / 恐怖 {summary.get('average_fear', '-')} / 連帯 {summary.get('average_solidarity', '-')}",
                f"SNS投稿数: {summary.get('total_posts', '-')} / 誤情報系: {summary.get('false_posts', 0) + summary.get('misleading_posts', 0)} / 訂正: {summary.get('correction_posts', 0)}",
                f"LLM判断ログ: {source_counts.get('llm', 0)}件",
            ]
        else:
            result_items = ["LLMデモ実行後に summary.json と decisions.jsonl から自動反映"]
        bullet_list(ax, result_items, size=14, width=64)
        stats_path = output_dir / "statistics.png"
        if stats_path.exists():
            img = plt.imread(stats_path)
            ax.imshow(img, extent=(0.50, 0.95, 0.14, 0.64), aspect="auto")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = setup_slide("考察と限界")
        bullet_list(
            ax,
            [
                "大規模な抗議参加が自然発生するところまでは観察できなかった",
                "興味深い点は、怒り・恐怖・連帯感が上がっても、行動はまず確認に向かったこと",
                "日常会話が、イベント後に「噂を確かめたい」という同時行動へ変化した",
                "創発は暴動化ではなく、不安な確認行動の同期として現れた",
                "要因として、市民10人・50ステップ、公式訂正、行動候補の制約、LLMの慎重さが考えられる",
                "次の課題は、社会ネットワーク構造と同調圧力をより明示すること",
            ],
            size=14,
            width=62,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = setup_slide("技術的特徴")
        bullet_list(
            ax,
            [
                "Mesa / Solara を使わない軽量なPython実装",
                "LLM出力はJSONに限定し、検証後に状態へ反映",
                "判断・発話・SNS投稿をJSONLとして保存",
                "matplotlib と imageio でログ付き動画を生成",
            ],
            size=15,
            width=58,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = setup_slide("今後の展望")
        bullet_list(
            ax,
            [
                "公式訂正のタイミングと信頼度を変えて比較する",
                "SNS活動度やメディアリテラシーの分布を変える",
                "複数モデルでLLMエージェントの頑健性を検証する",
                "発話履歴と空間挙動をインタラクティブに探索する",
            ],
            size=15,
            width=58,
        )
        pdf.savefig(fig)
        plt.close(fig)

    print(pdf_path)


if __name__ == "__main__":
    main()
