"""
Generate an MP4 video from simulation output frames.

Usage:
  python generate_video.py output/ -o result.mp4 --fps 10
  python generate_video.py output/ -o result.mp4 --fps 10 --with-log
  python generate_video.py output/ -o result.mp4 --fps 5 --width 1280

Options:
  --with-log   Embed decisions.jsonl/messages.jsonl as a right-side panel
  --width      Total output video width in pixels (default: 960 without log, 1440 with log)
  --fps        Frames per second (default: 10)
"""
import argparse
import glob
import json
import logging
import os
import textwrap
from collections import Counter, defaultdict
from typing import Optional

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PANEL_BG = (20, 20, 30)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (100, 200, 255)
STATE_TEXT_COLORS = {
    "participant": (255, 80, 80),
    "observer":    (150, 210, 240),
    "skeptic":     (255, 220, 50),
    "mediator":    (60, 200, 100),
    "withdrawn":   (180, 100, 220),
    "detained":    (160, 160, 160),
    "ordinary":    (160, 160, 160),
}

STATE_LABELS = {
    "participant": "抗議参加",
    "observer": "観察中",
    "skeptic": "懐疑的",
    "mediator": "仲裁",
    "withdrawn": "退避",
    "detained": "拘束",
    "ordinary": "通常",
}

SOURCE_LABELS = {
    "llm": "Ollama LLM",
    "fallback": "ルール補完",
    "llm_empty_fallback": "LLM空応答補完",
    "llm_parse_fallback": "LLM解析失敗補完",
}


def _try_font(size: int):
    """Load a Japanese-capable font if available, else default."""
    candidates = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _load_decisions(output_dir: str) -> dict[int, list[dict]]:
    """Load decisions.jsonl grouped by step."""
    path = os.path.join(output_dir, "decisions.jsonl")
    by_step: dict[int, list[dict]] = defaultdict(list)
    if not os.path.exists(path):
        return by_step
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                by_step[rec["step"]].append(rec)
            except Exception:
                pass
    return by_step


def _load_messages(output_dir: str) -> dict[int, list[dict]]:
    """Load messages.jsonl grouped by step."""
    path = os.path.join(output_dir, "messages.jsonl")
    by_step: dict[int, list[dict]] = defaultdict(list)
    if not os.path.exists(path):
        return by_step
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                by_step[rec["step"]].append(rec)
            except Exception:
                pass
    return by_step


def _load_social_posts(output_dir: str) -> dict[int, list[dict]]:
    """Load social_feed.jsonl grouped by step."""
    path = os.path.join(output_dir, "social_feed.jsonl")
    by_step: dict[int, list[dict]] = defaultdict(list)
    if not os.path.exists(path):
        return by_step
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                by_step[rec["step"]].append(rec)
            except Exception:
                pass
    return by_step


def _build_stats_by_step(
    decisions: dict[int, list[dict]],
    social_posts: dict[int, list[dict]],
) -> dict[int, dict]:
    """Build lightweight per-step/cumulative stats for the video panel."""
    stats: dict[int, dict] = {}
    cumulative_posts = 0
    cumulative_falseish = 0
    cumulative_corrections = 0
    all_steps = sorted(set(decisions.keys()) | set(social_posts.keys()))
    for step in all_steps:
        posts = social_posts.get(step, [])
        cumulative_posts += len(posts)
        cumulative_falseish += sum(
            1
            for p in posts
            if p.get("truth_status") in {"FALSE", "MISLEADING", "UNVERIFIED"}
        )
        cumulative_corrections += sum(
            1 for p in posts if p.get("truth_status") == "CORRECTION"
        )

        state_counts = Counter(d.get("public_state", "unknown") for d in decisions.get(step, []))
        stats[step] = {
            "participant": state_counts.get("participant", 0),
            "observer": state_counts.get("observer", 0),
            "skeptic": state_counts.get("skeptic", 0),
            "ordinary": state_counts.get("ordinary", 0),
            "withdrawn": state_counts.get("withdrawn", 0),
            "detained": state_counts.get("detained", 0),
            "llm_decisions": sum(1 for d in decisions.get(step, []) if d.get("decision_source") == "llm"),
            "total_posts": cumulative_posts,
            "falseish_posts": cumulative_falseish,
            "correction_posts": cumulative_corrections,
        }
    return stats


def _group_messages(msgs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for msg in msgs:
        state = msg.get("public_state", "ordinary")
        utterance = msg.get("utterance", "")
        key = (state, utterance)
        if key not in grouped:
            grouped[key] = {
                "public_state": state,
                "utterance": utterance,
                "agent_ids": [],
                "count": 0,
            }
        grouped[key]["agent_ids"].append(msg.get("agent_id"))
        grouped[key]["count"] += 1
    return list(grouped.values())


def _build_log_panel(
    step: int,
    decisions: dict[int, list[dict]],
    messages: dict[int, list[dict]],
    social_posts: dict[int, list[dict]],
    stats_by_step: dict[int, dict],
    panel_w: int,
    panel_h: int,
) -> Image.Image:
    """Build the right-side log panel for a given step."""
    img = Image.new("RGB", (panel_w, panel_h), PANEL_BG)
    draw = ImageDraw.Draw(img)

    font_title = _try_font(14)
    font_body = _try_font(11)
    font_small = _try_font(9)

    y = 10
    line_h_title = 18
    line_h_body = 14

    # Title
    draw.text((10, y), f"Step {step} — Agent Log", font=font_title, fill=ACCENT_COLOR)
    y += line_h_title + 4
    draw.line([(10, y), (panel_w - 10, y)], fill=(60, 60, 80), width=1)
    y += 6

    # Stats
    stats = stats_by_step.get(step, {})
    if stats:
        draw.text((10, y), "統計 (Stats)", font=font_body, fill=ACCENT_COLOR)
        y += line_h_body + 2
        lines = [
            f"LLMによる市民判断: {stats.get('llm_decisions', 0)} 件",
            f"市民状態: 抗議参加 {stats.get('participant', 0)} / 観察中 {stats.get('observer', 0)} / 懐疑的 {stats.get('skeptic', 0)}",
            f"市民状態: 退避 {stats.get('withdrawn', 0)} / 拘束 {stats.get('detained', 0)} / 通常 {stats.get('ordinary', 0)}",
            f"SNS投稿累計: {stats.get('total_posts', 0)} 件",
            f"噂・未確認投稿: {stats.get('falseish_posts', 0)} 件 / 訂正投稿: {stats.get('correction_posts', 0)} 件",
        ]
        for line in lines:
            draw.text((14, y), line, font=font_small, fill=(210, 210, 210))
            y += line_h_body
        y += 4
        draw.line([(10, y), (panel_w - 10, y)], fill=(40, 40, 60), width=1)
        y += 6

    # Social feed
    posts = social_posts.get(step, [])
    if posts:
        draw.text((10, y), "SNS投稿 (Social Feed)", font=font_body, fill=ACCENT_COLOR)
        y += line_h_body + 2
        for p in posts[:5]:
            status = p.get("truth_status", "")
            tone = p.get("emotional_tone", "")
            author = p.get("author_type", "")
            header = f"[{status}/{tone}] {author}"
            draw.text((14, y), header, font=font_small, fill=(255, 220, 120))
            y += line_h_body
            content = p.get("content", "")
            for chunk in textwrap.wrap(content, width=42)[:2]:
                if y > panel_h - 20:
                    break
                draw.text((18, y), chunk, font=font_small, fill=(210, 210, 210))
                y += line_h_body
            y += 2
        y += 4
        draw.line([(10, y), (panel_w - 10, y)], fill=(40, 40, 60), width=1)
        y += 6

    # Utterances
    msgs = messages.get(step, [])
    if msgs:
        draw.text((10, y), "💬 発言 (Utterances)", font=font_body, fill=ACCENT_COLOR)
        y += line_h_body + 2
        for m in _group_messages(msgs)[:5]:
            state = m.get("public_state", "ordinary")
            color = STATE_TEXT_COLORS.get(state, TEXT_COLOR)
            ids = ",".join(str(i) for i in m.get("agent_ids", []) if i is not None)
            count = m.get("count", 1)
            count_label = f" 同内容{count}人" if count > 1 else ""
            prefix = f"[{STATE_LABELS.get(state, state)} #{ids}{count_label}] "
            text = prefix + m.get("utterance", "")
            for chunk in textwrap.wrap(text, width=38):
                if y > panel_h - 20:
                    break
                draw.text((14, y), chunk, font=font_small, fill=color)
                y += line_h_body
        y += 4
        draw.line([(10, y), (panel_w - 10, y)], fill=(40, 40, 60), width=1)
        y += 6

    # Decisions (interpretation + action_reason)
    decs = decisions.get(step, [])
    if decs:
        draw.text((10, y), "🧠 解釈と判断 (Decisions)", font=font_body, fill=ACCENT_COLOR)
        y += line_h_body + 2
        shown = 0
        for d in decs:
            if shown >= 3:
                break
            interp = d.get("interpretation", "").strip()
            reason = d.get("action_reason", "").strip()
            intent = d.get("intent", "")
            state = d.get("public_state", "ordinary")
            source = SOURCE_LABELS.get(d.get("decision_source", "unknown"), d.get("decision_source", "unknown"))
            color = STATE_TEXT_COLORS.get(state, TEXT_COLOR)
            header = f"#{d['agent_id']} [{STATE_LABELS.get(state, state)}] {source} → {intent}"
            draw.text((14, y), header, font=font_small, fill=color)
            y += line_h_body
            if interp:
                for chunk in textwrap.wrap(interp, width=40)[:2]:
                    if y > panel_h - 20:
                        break
                    draw.text((18, y), chunk, font=font_small, fill=(180, 180, 180))
                    y += line_h_body
            if reason:
                for chunk in textwrap.wrap(f"→ {reason}", width=40)[:1]:
                    if y > panel_h - 20:
                        break
                    draw.text((18, y), chunk, font=font_small, fill=(140, 200, 140))
                    y += line_h_body
            y += 2
            shown += 1

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MP4 from simulation frames")
    parser.add_argument("output_dir", help="Path to output/ directory")
    parser.add_argument("-o", "--output", default="result.mp4", help="Output MP4 filename")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--with-log", action="store_true", help="Add log panel on the right")
    parser.add_argument("--width", type=int, default=None, help="Output video width in pixels")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = args.output_dir.rstrip("/\\")
    frame_paths = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))

    if not frame_paths:
        logger.error(f"frame_*.png が見つかりません: {output_dir}")
        return

    logger.info(f"{len(frame_paths)} フレームを検出しました。")

    decisions_by_step: dict = {}
    messages_by_step: dict = {}
    social_posts_by_step: dict = {}
    stats_by_step: dict = {}

    if args.with_log:
        decisions_by_step = _load_decisions(output_dir)
        messages_by_step = _load_messages(output_dir)
        social_posts_by_step = _load_social_posts(output_dir)
        stats_by_step = _build_stats_by_step(decisions_by_step, social_posts_by_step)

    # Determine output dimensions from first frame
    sample = Image.open(frame_paths[0])
    frame_w, frame_h = sample.size
    sample.close()

    if args.with_log:
        log_panel_w = args.width - frame_w if args.width else max(400, frame_w // 2)
        total_w = frame_w + log_panel_w
        total_h = frame_h
    else:
        total_w = args.width or frame_w
        total_h = frame_h

    logger.info(f"出力サイズ: {total_w}x{total_h}  FPS: {args.fps}")

    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )

    for frame_path in frame_paths:
        # Extract step number from filename (frame_0010.png → 10)
        basename = os.path.basename(frame_path)
        try:
            step_num = int(basename.replace("frame_", "").replace(".png", ""))
        except ValueError:
            step_num = 0

        frame_img = Image.open(frame_path).convert("RGB")

        if frame_w != (total_w if not args.with_log else frame_w):
            frame_img = frame_img.resize((total_w if not args.with_log else frame_w, total_h),
                                         Image.LANCZOS)

        if args.with_log:
            log_img = _build_log_panel(
                step_num,
                decisions_by_step,
                messages_by_step,
                social_posts_by_step,
                stats_by_step,
                log_panel_w, total_h,
            )
            combined = Image.new("RGB", (total_w, total_h))
            combined.paste(frame_img, (0, 0))
            combined.paste(log_img, (frame_w, 0))
            arr = np.array(combined)
        else:
            arr = np.array(frame_img)

        # Ensure even dimensions for H.264 encoding
        h, w = arr.shape[:2]
        if h % 2 != 0 or w % 2 != 0:
            arr = arr[:h - (h % 2), :w - (w % 2)]

        writer.append_data(arr)

    writer.close()
    logger.info(f"動画を保存しました: {args.output}")
    logger.info(f"再生コマンド: start {args.output}  (Windows) / open {args.output}  (Mac)")


if __name__ == "__main__":
    main()
