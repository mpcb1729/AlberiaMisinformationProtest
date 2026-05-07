"""
Matplotlib-based visualization for Misinformation Protest simulation.
Saves per-step PNG frames and a final statistics plot.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from simulation import Simulation

logger = logging.getLogger(__name__)

# Use non-interactive Agg backend for frame saving (no display required)
matplotlib.use("Agg")

# Configure Japanese font if available (Windows: Meiryo, Mac: Hiragino)
def _setup_japanese_font() -> None:
    import matplotlib.font_manager as fm
    candidates = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                prop = fm.FontProperties(fname=path)
                plt.rcParams["font.family"] = prop.get_name()
                plt.rcParams["axes.unicode_minus"] = False
                logger.debug(f"Japanese font loaded: {path}")
                return
            except Exception:
                pass

_setup_japanese_font()

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

STATE_COLORS = {
    "ordinary":    "#888888",
    "observer":    "#ADD8E6",
    "participant": "#FF4444",
    "mediator":    "#2ECC40",
    "skeptic":     "#FFD700",
    "withdrawn":   "#9B59B6",
    "detained":    "#222222",
}

STATE_LABELS = {
    "ordinary":    "通常 (Ordinary)",
    "observer":    "観察 (Observer)",
    "participant": "参加 (Participant)",
    "mediator":    "仲裁 (Mediator)",
    "skeptic":     "懐疑 (Skeptic)",
    "withdrawn":   "退避 (Withdrawn)",
    "detained":    "拘束 (Detained)",
}

COP_COLOR        = "#003366"
INFLUENCER_COLOR = "#FF69B4"

# Landmark background colors (RGBA)
LANDMARK_COLORS: dict[str, tuple] = {
    "home_area":         (0.85, 0.95, 0.85, 0.3),
    "station":           (0.95, 0.95, 0.75, 0.3),
    "central_square":    (1.00, 0.90, 0.85, 0.35),
    "city_hall":         (0.90, 0.85, 1.00, 0.35),
    "media_zone":        (0.85, 0.95, 1.00, 0.3),
    "small_gathering_a": (1.00, 1.00, 0.85, 0.25),
    "small_gathering_b": (1.00, 1.00, 0.85, 0.25),
    "police_station":    (0.85, 0.85, 0.95, 0.35),
    "exit_zone":         (0.95, 0.85, 0.85, 0.25),
}

LANDMARK_SHORT_LABELS: dict[str, str] = {
    "home_area":         "Home",
    "station":           "Station",
    "central_square":    "Central Sq.",
    "city_hall":         "City Hall",
    "media_zone":        "Media",
    "small_gathering_a": "Gather A",
    "small_gathering_b": "Gather B",
    "police_station":    "Police",
    "exit_zone":         "Exit",
}


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class Visualizer:
    """Handles per-step frame rendering and final statistics plot."""

    def __init__(self, fig_size: tuple = (12, 10), dpi: int = 120) -> None:
        self.fig_size = fig_size
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Per-step frame
    # ------------------------------------------------------------------

    def visualize_step(
        self,
        sim: "Simulation",
        step: int,
        save_path: Optional[str] = None,
    ) -> None:
        """Render the current grid state and optionally save as PNG."""
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)

        # Background: landmark regions
        self._draw_landmarks(ax, sim)

        from agents import CitizenAgent, CopAgent, InfluencerAgent

        # Pre-compute per-agent jittered positions (so ID label matches dot)
        citizen_plot: list[dict] = []   # {x, y, id, state, utterance, pos}
        cop_plot: list[dict] = []
        influencer_plot: list[dict] = []

        rng = np.random.default_rng(sim.steps)   # deterministic per step for reproducibility
        for agents_at_cell in sim.grid.values():
            for agent in agents_at_cell:
                if isinstance(agent, CitizenAgent):
                    jx = agent.pos[0] + rng.uniform(-0.3, 0.3)
                    jy = agent.pos[1] + rng.uniform(-0.3, 0.3)
                    citizen_plot.append({
                        "x": jx, "y": jy,
                        "id": agent.id,
                        "state": agent.public_state.value,
                        "utterance": getattr(agent, "utterance", ""),
                        "pos": agent.pos,
                    })
                elif isinstance(agent, CopAgent):
                    jx = agent.pos[0] + rng.uniform(-0.2, 0.2)
                    jy = agent.pos[1] + rng.uniform(-0.2, 0.2)
                    cop_plot.append({"x": jx, "y": jy, "id": agent.id, "pos": agent.pos})
                elif isinstance(agent, InfluencerAgent):
                    jx = agent.pos[0] + rng.uniform(-0.2, 0.2)
                    jy = agent.pos[1] + rng.uniform(-0.2, 0.2)
                    influencer_plot.append({"x": jx, "y": jy, "id": agent.id, "pos": agent.pos})

        # --- Conversation connectors ---
        # Two citizens are "conversing" when both are within vision radius and at
        # least one of them spoke this step (has a non-empty utterance).
        vision = sim.config.get("agents", {}).get("vision", 3)
        speakers = {d["id"] for d in citizen_plot if d["utterance"]}
        if speakers:
            pos_by_id = {d["id"]: d for d in citizen_plot}
            drawn_pairs: set[tuple] = set()
            for d in citizen_plot:
                if d["id"] not in speakers and not d["utterance"]:
                    continue
                px, py = d["pos"]
                for d2 in citizen_plot:
                    if d2["id"] <= d["id"]:
                        continue
                    pair = (min(d["id"], d2["id"]), max(d["id"], d2["id"]))
                    if pair in drawn_pairs:
                        continue
                    # Chebyshev distance check
                    qx, qy = d2["pos"]
                    if abs(qx - px) <= vision and abs(qy - py) <= vision:
                        if d["utterance"] or d2["utterance"]:
                            ax.plot(
                                [d["x"], d2["x"]], [d["y"], d2["y"]],
                                color="#aaaaaa", linewidth=0.8, alpha=0.5,
                                zorder=2, solid_capstyle="round",
                            )
                            drawn_pairs.add(pair)

        # --- Scatter citizens grouped by state ---
        citizen_by_state: dict[str, list[dict]] = {s: [] for s in STATE_COLORS}
        for d in citizen_plot:
            citizen_by_state.setdefault(d["state"], []).append(d)

        for state_val, agents in citizen_by_state.items():
            if not agents:
                continue
            ax.scatter(
                [d["x"] for d in agents], [d["y"] for d in agents],
                c=STATE_COLORS[state_val], s=80, alpha=0.85,
                edgecolors="white", linewidths=0.5, zorder=3,
            )
            # Agent ID labels
            for d in agents:
                ax.text(
                    d["x"], d["y"] + 0.22, str(d["id"]),
                    fontsize=4.5, ha="center", va="bottom",
                    color="#222222", zorder=5,
                )

        # --- Cops ---
        for d in cop_plot:
            ax.scatter(d["x"], d["y"], c=COP_COLOR, s=120, marker="^", alpha=0.9,
                       edgecolors="white", linewidths=0.5, zorder=4)
            ax.text(d["x"], d["y"] + 0.25, f"C{d['id']}",
                    fontsize=4.5, ha="center", va="bottom", color=COP_COLOR, zorder=5)
        if cop_plot:
            ax.scatter([], [], c=COP_COLOR, s=120, marker="^",
                       label="安全担当官 (Cop)")

        # --- Influencers ---
        for d in influencer_plot:
            ax.scatter(d["x"], d["y"], c=INFLUENCER_COLOR, s=150, marker="*",
                       alpha=0.95, edgecolors="white", linewidths=0.5, zorder=4)
            ax.text(d["x"], d["y"] + 0.28, f"I{d['id']}",
                    fontsize=4.5, ha="center", va="bottom", color=INFLUENCER_COLOR, zorder=5)
        if influencer_plot:
            ax.scatter([], [], c=INFLUENCER_COLOR, s=150, marker="*",
                       label="インフルエンサー")

        # Grid lines
        ax.set_xlim(-0.5, sim.width - 0.5)
        ax.set_ylim(-0.5, sim.height - 0.5)
        ax.set_xticks(np.arange(-0.5, sim.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, sim.height, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=0.3, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # Legend (citizens)
        legend_patches = [
            mpatches.Patch(color=STATE_COLORS[s], label=STATE_LABELS[s])
            for s in STATE_COLORS
        ]
        legend_patches.append(mpatches.Patch(color=COP_COLOR, label="安全担当官 (Cop)"))
        legend_patches.append(mpatches.Patch(color=INFLUENCER_COLOR, label="インフルエンサー"))
        ax.legend(handles=legend_patches, loc="upper left", fontsize=7,
                  framealpha=0.8, ncol=2)

        # Statistics in title
        if sim.stats_history:
            s = sim.stats_history[-1]
            title_stats = (
                f"参加:{s.get('participant_count',0)} "
                f"観察:{s.get('observer_count',0)} "
                f"懐疑:{s.get('skeptic_count',0)} "
                f"退避:{s.get('withdrawn_count',0)} "
                f"拘束:{s.get('detained_count',0)} "
                f"| 投稿:{s.get('total_posts',0)} "
                f"(誤:{s.get('false_posts',0)}+誤導:{s.get('misleading_posts',0)}"
                f" 訂正:{s.get('correction_posts',0)})"
            )
        else:
            title_stats = ""

        ax.set_title(
            f"Misinformation Protest — Step {step}\n{title_stats}",
            fontsize=10, pad=8,
        )
        ax.set_aspect("equal")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.debug(f"Frame saved: {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def _draw_landmarks(self, ax, sim: "Simulation") -> None:
        """Draw colored landmark rectangles with short labels."""
        for name, lm in sim.landmarks.items():
            x0, x1 = lm["x"]
            y0, y1 = lm["y"]
            color = LANDMARK_COLORS.get(name, (0.9, 0.9, 0.9, 0.2))
            rect = mpatches.FancyBboxPatch(
                (x0 - 0.5, y0 - 0.5),
                x1 - x0 + 1,
                y1 - y0 + 1,
                boxstyle="round,pad=0.05",
                facecolor=color[:3],
                alpha=color[3],
                edgecolor="gray",
                linewidth=0.8,
                zorder=1,
            )
            ax.add_patch(rect)
            cx, cy = lm["center"]
            label = LANDMARK_SHORT_LABELS.get(name, name)
            ax.text(cx, y0 - 0.3, label, fontsize=6, ha="center", va="top",
                    color="gray", style="italic", zorder=2)

    # ------------------------------------------------------------------
    # Statistics plot
    # ------------------------------------------------------------------

    def plot_statistics(
        self,
        stats_history: list[dict],
        save_path: Optional[str] = None,
    ) -> None:
        """4-panel statistics plot over all steps."""
        if not stats_history:
            return

        steps = [s["step"] for s in stats_history]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
        fig.suptitle("Misinformation Protest — Simulation Statistics", fontsize=13, y=1.01)

        # Panel 1: State counts
        ax1 = axes[0, 0]
        for state_val, color in STATE_COLORS.items():
            vals = [s.get(f"{state_val}_count", 0) for s in stats_history]
            ax1.plot(steps, vals, label=STATE_LABELS[state_val], color=color, linewidth=1.5)
        ax1.set_title("市民の状態別人数")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Count")
        ax1.legend(fontsize=7, loc="upper right")
        ax1.grid(alpha=0.3)

        # Panel 2: Average emotions
        ax2 = axes[0, 1]
        emotion_series = {
            "怒り (anger)":     ("#E74C3C", "average_anger"),
            "恐怖 (fear)":      ("#9B59B6", "average_fear"),
            "連帯 (solidarity)": ("#27AE60", "average_solidarity"),
            "噂信念 (belief)":   ("#F39C12", "average_belief"),
        }
        for label, (color, key) in emotion_series.items():
            vals = [s.get(key, 0) for s in stats_history]
            ax2.plot(steps, vals, label=label, color=color, linewidth=1.5)
        ax2.set_title("感情・信念の平均値")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Value (0–1)")
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # Panel 3: Post counts
        ax3 = axes[1, 0]
        for key, label, color in [
            ("false_posts",       "虚偽 (FALSE)",      "#E74C3C"),
            ("misleading_posts",  "誤導 (MISLEADING)", "#E67E22"),
            ("correction_posts",  "訂正 (CORRECTION)", "#27AE60"),
            ("unverified_posts",  "未確認 (UNVERIFIED)", "#95A5A6"),
        ]:
            vals = [s.get(key, 0) for s in stats_history]
            ax3.plot(steps, vals, label=label, color=color, linewidth=1.5)
        ax3.set_title("SNS投稿数（種類別）")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Post count")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        # Panel 4: Location density
        ax4 = axes[1, 1]
        cs_vals = [s.get("central_square_density", 0) for s in stats_history]
        ch_vals = [s.get("city_hall_density", 0) for s in stats_history]
        ax4.plot(steps, cs_vals, label="Central Square", color="#FF4444", linewidth=1.5)
        ax4.plot(steps, ch_vals, label="City Hall", color="#9B59B6", linewidth=1.5)
        ax4.set_title("ランドマーク密度")
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Density (agents/cell)")
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)

        # Add vertical lines for event steps
        for ax in [ax1, ax2, ax3, ax4]:
            for step_val, label in [(15, "evt1"), (40, "evt2"), (65, "evt3")]:
                if step_val <= max(steps, default=0):
                    ax.axvline(x=step_val, color="gray", linestyle="--",
                               linewidth=0.8, alpha=0.6)
                    ax.text(step_val + 0.5, ax.get_ylim()[1] * 0.95, label,
                            fontsize=6, color="gray", va="top")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Statistics plot saved: {save_path}")
        else:
            plt.show()

        plt.close(fig)
