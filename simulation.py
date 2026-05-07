"""
Simulation class for Misinformation Protest.
Manages the grid, agents, events, step execution, and output logging.
"""
from __future__ import annotations

import json
import logging
import os
import random
import uuid
from collections import defaultdict
from typing import Optional

import yaml

from agents import (
    CitizenAgent,
    CopAgent,
    InfluencerAgent,
    OfficialAgent,
    PublicState,
    SocialFeed,
    SocialPost,
)
from ollama_client import OllamaClient

logger = logging.getLogger(__name__)

LOG_INTERVAL = 10


class Simulation:
    """
    Core simulation engine.

    Grid: width × height cells.
    self.grid: dict[(x, y) → list[agent]] — supports multiple agents per cell.
    """

    def __init__(self, config: dict, output_dir: Optional[str] = None) -> None:
        self.config = config
        self.output_dir = output_dir

        sim_cfg = config.get("simulation", {})
        self.width: int = sim_cfg.get("width", 20)
        self.height: int = sim_cfg.get("height", 20)
        self.total_steps: int = sim_cfg.get("steps", 100)
        seed: int = sim_cfg.get("seed", 42)
        random.seed(seed)

        agents_cfg = config.get("agents", {})
        llm_cfg = config.get("llm", {})
        self.use_llm: bool = llm_cfg.get("use_llm", False)

        # Parse landmarks
        self.landmarks: dict = {}
        for name, lm in config.get("landmarks", {}).items():
            self.landmarks[name] = {
                "x": lm["x"],
                "y": lm["y"],
                "center": lm["center"],
            }

        # Parse events
        self.events: list[dict] = config.get("events", [])
        self.triggered_events: set[str] = set()
        self.event_rumors: dict[str, str] = {e["event_id"]: e.get("rumor", "") for e in self.events}
        self.event_truths: dict[str, str] = {e["event_id"]: e.get("ground_truth", "") for e in self.events}

        # Social feed
        self.social_feed = SocialFeed()

        # Grid: pos → [agents]
        self.grid: dict[tuple[int, int], list] = defaultdict(list)

        # Agent lists
        self.citizens: list[CitizenAgent] = []
        self.cops: list[CopAgent] = []
        self.influencers: list[InfluencerAgent] = []
        self.officials: list[OfficialAgent] = []
        self._next_id = 0

        # Step counter
        self.steps = 0

        # Statistics history
        self.stats_history: list[dict] = []

        # LLM client
        self.llm_client: Optional[OllamaClient] = None
        if self.use_llm:
            self.llm_client = OllamaClient(
                base_url=llm_cfg.get("base_url", "http://localhost:11434"),
                model=llm_cfg.get("model", "qwen3:8b"),
                temperature=llm_cfg.get("temperature", 0.4),
                max_tokens=llm_cfg.get("max_tokens", 4096),
                repeat_penalty=llm_cfg.get("repeat_penalty", 1.1),
                repeat_last_n=llm_cfg.get("repeat_last_n", 128),
                min_p=llm_cfg.get("min_p", 0.05),
            )

        # JSONL output handles (opened lazily)
        self._decisions_fh = None
        self._messages_fh = None
        self._social_feed_fh = None
        self._social_feed_written: set[str] = set()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._decisions_fh = open(os.path.join(output_dir, "decisions.jsonl"), "w", encoding="utf-8")
            self._messages_fh = open(os.path.join(output_dir, "messages.jsonl"), "w", encoding="utf-8")
            self._social_feed_fh = open(os.path.join(output_dir, "social_feed.jsonl"), "w", encoding="utf-8")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_agents(self) -> None:
        """Place all agents on the grid."""
        agents_cfg = self.config.get("agents", {})
        n_citizens = max(1, int(self.width * self.height * agents_cfg.get("citizen_density", 0.10)))
        n_cops = max(1, int(self.width * self.height * agents_cfg.get("cop_density", 0.02)))
        n_influencers = agents_cfg.get("influencer_count", 3)
        n_officials = agents_cfg.get("official_count", 1)

        # Citizens — bias toward home_area
        home = self.landmarks.get("home_area", {})
        for _ in range(n_citizens):
            if random.random() < 0.6 and home:
                pos = self._random_pos_in_landmark(home)
            else:
                pos = self._random_pos()
            agent = CitizenAgent(
                agent_id=self._next_id,
                pos=pos,
                config=self.config,
                llm_client=self.llm_client if self.use_llm else None,
            )
            self._next_id += 1
            self.citizens.append(agent)
            self.grid[pos].append(agent)

        # Cops — near police_station
        police_lm = self.landmarks.get("police_station", {})
        for _ in range(n_cops):
            if police_lm:
                pos = self._random_pos_in_landmark(police_lm)
            else:
                pos = self._random_pos()
            cop = CopAgent(
                agent_id=self._next_id,
                pos=pos,
                config=self.config,
            )
            self._next_id += 1
            self.cops.append(cop)
            self.grid[pos].append(cop)

        # Influencers — near media_zone
        media_lm = self.landmarks.get("media_zone", {})
        influencer_cfgs = self.config.get("influencers", [])
        for i in range(n_influencers):
            if media_lm:
                pos = self._random_pos_in_landmark(media_lm)
            else:
                pos = self._random_pos()
            icfg = influencer_cfgs[i] if i < len(influencer_cfgs) else {}
            inf = InfluencerAgent(
                agent_id=self._next_id,
                pos=pos,
                alignment=icfg.get("alignment", "sensational"),
                reach_base=icfg.get("reach_base", 20),
                posting_interval=icfg.get("posting_interval", 5),
            )
            self._next_id += 1
            self.influencers.append(inf)
            self.grid[pos].append(inf)

        # Officials — no grid position needed
        official_cfg = self.config.get("official", {})
        for _ in range(n_officials):
            off = OfficialAgent(
                agent_id=self._next_id,
                response_delay=official_cfg.get("response_delay", 30),
                credibility=0.85,
            )
            self._next_id += 1
            self.officials.append(off)

        logger.info(
            f"Agents: {len(self.citizens)} citizens, {len(self.cops)} cops, "
            f"{len(self.influencers)} influencers, {len(self.officials)} officials"
        )

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step_simulation(self) -> None:
        self.steps += 1

        self._trigger_events()

        # Shuffle citizens (random order each step)
        random.shuffle(self.citizens)
        all_agents = self.citizens + self.cops + self.influencers + self.officials
        for agent in all_agents:
            agent.step(self)

        self._collect_stats()
        self._log_decisions()
        self._log_messages()
        self._flush_social_feed_log()

        if self.steps % LOG_INTERVAL == 0:
            s = self.stats_history[-1]
            logger.info(
                f"Step {self.steps}: participant={s.get('participant_count',0)} "
                f"observer={s.get('observer_count',0)} "
                f"detained={s.get('detained_count',0)} "
                f"posts={s.get('total_posts',0)}"
            )

    # ------------------------------------------------------------------
    # Event triggering
    # ------------------------------------------------------------------

    def _trigger_events(self) -> None:
        for evt in self.events:
            eid = evt["event_id"]
            if eid in self.triggered_events:
                continue
            if self.steps != evt["step"]:
                continue

            self.triggered_events.add(eid)
            logger.info(f"Event triggered: {eid} at step {self.steps}")

            location_name = evt.get("location", "")
            lm = self.landmarks.get(location_name, {})
            center = tuple(lm.get("center", [10, 10]))
            radius = evt.get("spread_radius", 3)
            ground_truth = evt.get("ground_truth", "")
            rumor = evt.get("rumor", "")
            truth_status = evt.get("truth_status", "UNVERIFIED")
            severity = float(evt.get("severity", 0.5))
            online_reach = evt.get("online_reach", 20)

            # Notify nearby citizens of the rumor
            nearby = self.get_agents_in_radius(center, radius)
            for agent in nearby:
                if isinstance(agent, CitizenAgent):
                    existing = agent.belief_rumors.get(eid, 0.0)
                    initial = min(0.4 + random.random() * 0.2, 1.0 - existing)
                    # Media-literate citizens get lower initial belief
                    initial *= (1.0 - agent.media_literacy * 0.5)
                    agent.belief_rumors[eid] = max(existing, initial)
                    agent.memory.append(
                        f"（ステップ{self.steps}）{location_name}付近で「{ground_truth}」という話と、"
                        f"「{rumor}」という噂の両方を聞いた。"
                    )
                    confusion = 0.12 if truth_status in ("FALSE", "MISLEADING", "UNVERIFIED") else 0.04
                    agent.anger = min(
                        1.0,
                        max(agent.anger, confusion + severity * (0.24 + agent.hardship * 0.34)),
                    )
                    agent.fear = min(
                        1.0,
                        max(agent.fear, confusion + severity * (0.20 + agent.risk_aversion * 0.30)),
                    )
                    agent.solidarity = min(
                        1.0,
                        max(agent.solidarity, 0.08 + severity * (0.18 + agent.trust_neighbors * 0.18)),
                    )

            # Initial SNS post about the event
            if ground_truth:
                content = f"【速報】{ground_truth} 一方でSNSでは「{rumor}」という話も広がっている。"
            else:
                content = f"【情報】{rumor}"
            post = SocialPost(
                post_id=str(uuid.uuid4())[:8],
                step=self.steps,
                author_id=-1,
                author_type="event",
                content=content,
                rumor_id=eid,
                truth_status=truth_status,
                emotional_tone="urgent",
                reach=online_reach,
                credibility_score=0.3,
            )
            self.social_feed.add_post(post)

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def get_agents_in_radius(self, pos: tuple, radius: int) -> list:
        """Return all agents within Chebyshev (Moore) distance."""
        result = []
        cx, cy = int(pos[0]), int(pos[1])
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                cell = (cx + dx, cy + dy)
                result.extend(self.grid.get(cell, []))
        return result

    def move_agent(self, agent, new_pos: tuple[int, int]) -> None:
        old_pos = agent.pos
        if old_pos in self.grid:
            try:
                self.grid[old_pos].remove(agent)
            except ValueError:
                pass
        agent.pos = new_pos
        self.grid[new_pos].append(agent)

    def step_toward(self, from_pos: tuple, to_pos: tuple) -> tuple[int, int]:
        """One-step Moore-neighbourhood move toward target."""
        fx, fy = int(from_pos[0]), int(from_pos[1])
        tx, ty = int(to_pos[0]), int(to_pos[1])
        dx = 0 if tx == fx else (1 if tx > fx else -1)
        dy = 0 if ty == fy else (1 if ty > fy else -1)
        nx = max(0, min(self.width - 1, fx + dx))
        ny = max(0, min(self.height - 1, fy + dy))
        return (nx, ny)

    def random_adjacent(self, pos: tuple) -> tuple[int, int]:
        """Random Moore-neighbour (including self)."""
        x, y = int(pos[0]), int(pos[1])
        dx, dy = random.randint(-1, 1), random.randint(-1, 1)
        nx = max(0, min(self.width - 1, x + dx))
        ny = max(0, min(self.height - 1, y + dy))
        return (nx, ny)

    def get_landmark_at(self, pos: tuple) -> Optional[str]:
        x, y = int(pos[0]), int(pos[1])
        for name, lm in self.landmarks.items():
            x0, x1 = lm["x"]
            y0, y1 = lm["y"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return None

    # ------------------------------------------------------------------
    # Placement helpers
    # ------------------------------------------------------------------

    def _random_pos(self) -> tuple[int, int]:
        return (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def _random_pos_in_landmark(self, lm: dict) -> tuple[int, int]:
        x0, x1 = lm["x"]
        y0, y1 = lm["y"]
        return (random.randint(x0, min(x1, self.width - 1)),
                random.randint(y0, min(y1, self.height - 1)))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _collect_stats(self) -> None:
        state_counts = {s.value: 0 for s in PublicState}
        total_anger = total_fear = total_solidarity = 0.0
        total_belief = 0.0
        n = len(self.citizens)

        for c in self.citizens:
            state_counts[c.public_state.value] += 1
            total_anger += c.anger
            total_fear += c.fear
            total_solidarity += c.solidarity
            total_belief += max(c.belief_rumors.values(), default=0.0)

        central_sq = self.landmarks.get("central_square", {})
        city_hall = self.landmarks.get("city_hall", {})

        def density_at(lm: dict) -> float:
            if not lm:
                return 0.0
            x0, x1 = lm["x"]
            y0, y1 = lm["y"]
            count = sum(
                len([a for a in self.grid.get((x, y), []) if isinstance(a, CitizenAgent)])
                for x in range(x0, x1 + 1)
                for y in range(y0, y1 + 1)
            )
            area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
            return count / area

        snap = {
            "step": self.steps,
            **{f"{k}_count": v for k, v in state_counts.items()},
            "average_anger": round(total_anger / max(1, n), 3),
            "average_fear": round(total_fear / max(1, n), 3),
            "average_solidarity": round(total_solidarity / max(1, n), 3),
            "average_belief": round(total_belief / max(1, n), 3),
            "total_posts": len(self.social_feed.posts),
            "false_posts": self.social_feed.count_by_truth_status("FALSE"),
            "misleading_posts": self.social_feed.count_by_truth_status("MISLEADING"),
            "correction_posts": self.social_feed.count_by_truth_status("CORRECTION"),
            "unverified_posts": self.social_feed.count_by_truth_status("UNVERIFIED"),
            "central_square_density": round(density_at(central_sq), 3),
            "city_hall_density": round(density_at(city_hall), 3),
        }
        self.stats_history.append(snap)

    def get_statistics(self) -> dict:
        if not self.stats_history:
            return {}
        final = self.stats_history[-1]
        max_participants = max(
            (s.get("participant_count", 0) for s in self.stats_history), default=0
        )
        peak_step = next(
            (s["step"] for s in self.stats_history
             if s.get("participant_count", 0) == max_participants), 0
        )
        return {
            **final,
            "peak_participant_count": max_participants,
            "peak_participant_step": peak_step,
            "total_steps_run": self.steps,
        }

    # ------------------------------------------------------------------
    # Output logging
    # ------------------------------------------------------------------

    def _log_decisions(self) -> None:
        if not self._decisions_fh:
            return
        for c in self.citizens:
            if not c.last_decision:
                continue
            record = {
                "step": self.steps,
                "agent_id": c.id,
                "pos": list(c.pos),
                "public_state": c.public_state.value,
                "thinking": c.last_decision.get("thinking", ""),
                "interpretation": c.last_decision.get("interpretation", ""),
                "belief_updates": c.last_decision.get("belief_updates", []),
                "emotion_update": c.last_decision.get("emotion_update", {}),
                "intent": c.current_intent,
                "target": c.current_target,
                "action_reason": c.last_decision.get("action_reason", ""),
                "utterance": c.utterance,
                "memory_to_add": c.last_decision.get("memory_to_add", ""),
                "decision_source": c.last_decision.get("_decision_source", "unknown"),
            }
            self._decisions_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._decisions_fh.flush()

    def _log_messages(self) -> None:
        if not self._messages_fh:
            return
        for c in self.citizens:
            if c.utterance:
                record = {
                    "step": self.steps,
                    "agent_id": c.id,
                    "pos": list(c.pos),
                    "public_state": c.public_state.value,
                    "utterance": c.utterance,
                }
                self._messages_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._messages_fh.flush()

    def _flush_social_feed_log(self) -> None:
        if not self._social_feed_fh:
            return
        for post in self.social_feed.posts:
            if post.post_id in self._social_feed_written:
                continue
            record = {
                "post_id": post.post_id,
                "step": post.step,
                "author_id": post.author_id,
                "author_type": post.author_type,
                "content": post.content,
                "rumor_id": post.rumor_id,
                "truth_status": post.truth_status,
                "emotional_tone": post.emotional_tone,
                "reach": post.reach,
                "credibility_score": post.credibility_score,
            }
            self._social_feed_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._social_feed_written.add(post.post_id)
        self._social_feed_fh.flush()

    def save_summary(self) -> None:
        """Write summary.json to output_dir."""
        if not self.output_dir:
            return
        stats = self.get_statistics()
        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved: {path}")

    def close(self) -> None:
        """Close all open file handles."""
        for fh in (self._decisions_fh, self._messages_fh, self._social_feed_fh):
            if fh:
                fh.close()

    # ------------------------------------------------------------------
    # Ollama check
    # ------------------------------------------------------------------

    def check_ollama_setup(self) -> bool:
        """Check connection and model availability. Returns False if setup fails."""
        if not self.llm_client:
            return True

        if not self.llm_client.check_connection():
            logger.error(
                "Ollama に接続できません。\n"
                f"  期待する URL: {self.llm_client.base_url}\n"
                "  Ollama を起動してから再実行してください: https://ollama.com"
            )
            return False

        if not self.llm_client.check_model_exists():
            available = self.llm_client.list_models()
            model_name = self.llm_client.model
            logger.error(f"モデル '{model_name}' が見つかりません。")
            if available:
                logger.error(f"  利用可能なモデル: {', '.join(available)}")
            logger.error(f"  モデルをダウンロードしてください: ollama pull {model_name}")
            return False

        logger.info(f"Ollama OK: model={self.llm_client.model}")
        return True
