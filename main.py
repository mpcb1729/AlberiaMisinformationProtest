"""
Misinformation Protest Simulation — Entry Point

Usage:
  python main.py                          # run with config.yaml defaults (LLM on)
  python main.py --no-llm                 # rule-based fallback, no Ollama needed
  python main.py --save-frames            # save PNG frames to output/
  python main.py --steps 30              # override step count
  python main.py --config custom.yaml    # use a different config file
"""
import argparse
import logging
import os
import shutil
import time

import yaml


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"), logging.INFO)
    log_file = log_cfg.get("log_file", "simulation.log")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Misinformation Protest LLM Agent Simulation"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--save-frames", action="store_true", help="Save PNG frames to output/")
    parser.add_argument("--steps", type=int, default=None, help="Override step count from config")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (rule-based fallback)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.steps is not None:
        config["simulation"]["steps"] = args.steps
    if args.no_llm:
        config["llm"]["use_llm"] = False

    setup_logging(config)
    logger = logging.getLogger(__name__)

    vis_cfg = config.get("visualization", {})
    output_dir: str = vis_cfg.get("output_dir", "output")
    save_frames: bool = args.save_frames or vis_cfg.get("save_frames", False)
    frame_interval: int = vis_cfg.get("frame_interval", 1)

    # Clean and recreate output directory
    if save_frames:
        if os.path.exists(output_dir):
            logger.info(f"既存の出力ディレクトリを削除します: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        logger.info(f"出力ディレクトリ: {output_dir}")

    # Import here so logging is already set up
    from simulation import Simulation
    from visualization import Visualizer

    sim = Simulation(config=config, output_dir=output_dir if save_frames else None)
    sim.initialize_agents()

    use_llm = config.get("llm", {}).get("use_llm", False)
    if use_llm:
        if not sim.check_ollama_setup():
            logger.error("Ollama のセットアップに問題があります。--no-llm オプションで再試行するか、Ollama を起動してください。")
            sim.close()
            return
        logger.info(f"LLM モード: {sim.llm_client.model}")
    else:
        logger.info("ルールベースモード（LLM無効）")

    visualizer = Visualizer(
        fig_size=tuple(vis_cfg.get("figure_size", [12, 10])),
        dpi=vis_cfg.get("dpi", 120),
    ) if save_frames else None

    total_steps = config["simulation"]["steps"]
    logger.info(f"シミュレーション開始: {total_steps} ステップ")
    start_time = time.time()

    try:
        for step in range(1, total_steps + 1):
            sim.step_simulation()

            if visualizer and (step % frame_interval == 0 or step == total_steps):
                frame_path = os.path.join(output_dir, f"frame_{step:04d}.png")
                visualizer.visualize_step(sim, step, save_path=frame_path)

    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました。")
    except Exception as e:
        logger.error(f"シミュレーション中にエラーが発生しました: {e}", exc_info=True)
    finally:
        elapsed = time.time() - start_time
        logger.info(f"完了: {sim.steps} ステップ / {elapsed:.1f} 秒")

        if visualizer and sim.stats_history:
            stats_path = os.path.join(output_dir, "statistics.png")
            visualizer.plot_statistics(sim.stats_history, save_path=stats_path)
            logger.info(f"統計グラフ保存: {stats_path}")

        sim.save_summary()
        sim.close()

    # Print final summary
    stats = sim.get_statistics()
    sep = "=" * 40
    print(f"\n{sep}")
    print("=== Simulation Complete ===")
    print(f"{sep}")
    print(f"  Steps run      : {stats.get('total_steps_run', 0)}")
    print(f"  Peak participant: {stats.get('peak_participant_count', 0)} (Step {stats.get('peak_participant_step', 0)})")
    print(f"  Final - participant:{stats.get('participant_count', 0)}  observer:{stats.get('observer_count', 0)}  skeptic:{stats.get('skeptic_count', 0)}")
    print(f"  SNS posts total : {stats.get('total_posts', 0)}  (false:{stats.get('false_posts',0)+stats.get('misleading_posts',0)}  correction:{stats.get('correction_posts',0)})")
    if save_frames:
        print(f"\n  Output dir: {output_dir}/")
        print("    frame_*.png      - grid animation frames")
        print("    decisions.jsonl  - LLM decision log")
        print("    messages.jsonl   - utterance log")
        print("    social_feed.jsonl- SNS post history")
        print("    statistics.png   - stats chart")
        print("    summary.json     - final stats")
        print(f"\n  Generate video:")
        print(f"    python generate_video.py {output_dir}/ -o result.mp4 --fps 10")


if __name__ == "__main__":
    main()
