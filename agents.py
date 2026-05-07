"""
Agent definitions for the Misinformation Protest simulation.

Agents:
  CitizenAgent  - main LLM-driven agent with belief/emotion state
  CopAgent      - rule-based safety officer
  InfluencerAgent - periodic SNS poster with configurable alignment
  OfficialAgent   - delayed correction publisher

Also defines:
  PublicState   - visible state enum for visualization
  SocialPost    - immutable SNS post record
  SocialFeed    - shared post repository with sampling
"""
from __future__ import annotations

import json
import logging
import random
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from simulation import Simulation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIOLENCE_KEYWORDS = [
    "殺す", "殺せ", "爆弾", "放火", "火をつけ", "武器を持", "テロ",
    "kill ", "bomb", "weaponize", "terror",
]

UNPROMPTED_SUSPICION_KEYWORDS = [
    "何か変", "変なこと", "異変", "不審", "隠され", "何か起き",
]

VOICE_STYLES = [
    "短く率直。苛立つと語気が強くなる",
    "慎重で疑い深い。不安を遠回しに話す",
    "近所目線で生活感がある。家族や仕事への影響を気にする",
    "SNS慣れしている。焦ると短文で投稿したがる",
    "感情が表に出やすい。納得できない時は失望や怒りを隠さない",
    "周囲を気遣う。恐怖や混乱を感じても人に声をかける",
]

DAILY_TOPICS = [
    "駅前の混み具合",
    "配給所や店の待ち時間",
    "仕事帰りの疲れ",
    "近所の店の品ぞろえ",
    "家族や友人との予定",
    "天気と帰り道",
    "市庁舎の手続き",
    "地域メディアで見た生活情報",
]

VALID_INTENTS = [
    "stay_home",
    "observe",
    "ask_neighbor",
    "verify_rumor",
    "move_to_small_gathering",
    "move_to_central_square",
    "move_to_city_hall",
    "move_to_media_zone",
    "move_to_station",
    "retreat",
    "share_uncertain_info",
    "share_correction",
    "calm_others",
    "join_peaceful_protest",
]

# intent → target landmark mapping
INTENT_TO_LANDMARK: dict[str, str] = {
    "move_to_small_gathering": "small_gathering_a",
    "move_to_central_square": "central_square",
    "move_to_city_hall": "city_hall",
    "move_to_media_zone": "media_zone",
    "move_to_station": "station",
    "retreat": "exit_zone",
    "join_peaceful_protest": "central_square",
    "verify_rumor": "small_gathering_a",
    "share_correction": "media_zone",
    "calm_others": "central_square",
    "share_uncertain_info": "small_gathering_b",
    "ask_neighbor": None,
    "observe": None,
    "stay_home": None,
}

VALID_PUBLIC_STATES = {s.value for s in __import__("agents", fromlist=["PublicState"]).PublicState} \
    if False else None  # initialized after class definition


# ---------------------------------------------------------------------------
# PublicState
# ---------------------------------------------------------------------------

class PublicState(Enum):
    ORDINARY    = "ordinary"     # 通常生活
    OBSERVER    = "observer"     # 状況観察中
    PARTICIPANT = "participant"  # 集会・抗議参加
    MEDIATOR    = "mediator"     # 仲裁・沈静化
    SKEPTIC     = "skeptic"      # 情報を疑っている
    WITHDRAWN   = "withdrawn"    # 退避
    DETAINED    = "detained"     # 拘束中


VALID_PUBLIC_STATE_VALUES = {s.value for s in PublicState}


# ---------------------------------------------------------------------------
# SocialPost / SocialFeed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SocialPost:
    post_id: str
    step: int
    author_id: int
    author_type: str          # "citizen" | "influencer" | "official" | "cop_event" | "event"
    content: str
    rumor_id: str             # "" if not related to a specific event rumor
    truth_status: str         # TRUE / FALSE / UNVERIFIED / CORRECTION / MISLEADING
    emotional_tone: str       # calm / angry / fearful / urgent / skeptical
    reach: int
    credibility_score: float  # 0-1


class SocialFeed:
    """Shared SNS-like post repository."""

    def __init__(self) -> None:
        self.posts: list[SocialPost] = []

    def add_post(self, post: SocialPost) -> None:
        self.posts.append(post)

    def sample_posts(self, n: int, sns_activity: float) -> list[SocialPost]:
        """Sample up to n posts weighted by reach and sns_activity."""
        if not self.posts:
            return []
        k = min(n, len(self.posts))
        weights = [max(1, p.reach) for p in self.posts]
        return random.choices(self.posts, weights=weights, k=k)

    def count_by_truth_status(self, status: str) -> int:
        return sum(1 for p in self.posts if p.truth_status == status)

    def total_reach_by_status(self, status: str) -> int:
        return sum(p.reach for p in self.posts if p.truth_status == status)

    def recent_posts(self, n: int = 20) -> list[SocialPost]:
        return self.posts[-n:]


# ---------------------------------------------------------------------------
# CitizenAgent
# ---------------------------------------------------------------------------

class CitizenAgent:
    """
    LLM-driven citizen with belief/emotion state.
    World constraints (movement, action space) are enforced by Python;
    interpretation, belief updates, emotions, and intent are decided by the LLM.
    """

    def __init__(
        self,
        agent_id: int,
        pos: tuple[int, int],
        config: dict,
        llm_client=None,
    ) -> None:
        self.id = agent_id
        self.pos = pos
        self.llm_client = llm_client
        self._cfg = config

        agents_cfg = config.get("agents", {})
        self.vision: int = agents_cfg.get("vision", 3)
        self.memory_limit: int = agents_cfg.get("memory_limit", 10)
        self.message_context_size: int = agents_cfg.get("message_context_size", 5)
        self.post_context_size: int = agents_cfg.get("post_context_size", 5)
        self.llm_decision_interval: int = agents_cfg.get("llm_decision_interval", 1)
        use_llm_cfg: bool = config.get("llm", {}).get("use_llm", False)
        self.use_llm: bool = use_llm_cfg and (llm_client is not None)

        # Fixed personality (set once, never changed by LLM)
        self.hardship: float = random.random()
        self.risk_aversion: float = random.random()
        avg_ml = agents_cfg.get("average_media_literacy", 0.5)
        self.media_literacy: float = float(max(0.0, min(1.0, random.gauss(avg_ml, 0.15))))
        self.sns_activity: float = random.random()
        self.voice_style: str = VOICE_STYLES[agent_id % len(VOICE_STYLES)]
        self.daily_topic: str = DAILY_TOPICS[agent_id % len(DAILY_TOPICS)]

        # Mutable internal state (updated by LLM or fallback)
        self.public_state: PublicState = PublicState.ORDINARY
        self.anger: float = 0.0
        self.fear: float = 0.0
        self.solidarity: float = 0.0
        self.trust_official: float = float(agents_cfg.get("average_trust_in_official_info", 0.5))
        self.trust_neighbors: float = float(agents_cfg.get("average_trust_in_peer_info", 0.6))
        self.belief_rumors: dict[str, float] = {}  # rumor_id → strength 0-1
        self.private_stance: str = ""

        # Action state
        self.current_intent: str = "stay_home"
        self.current_target: Optional[str] = None
        self.utterance: str = ""
        self.memory: deque[str] = deque(maxlen=self.memory_limit)
        self.jail_sentence_left: int = 0
        self.step_counter: int = 0

        # For decisions log
        self.last_decision: dict = {}

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, sim: "Simulation") -> None:
        if self.public_state == PublicState.DETAINED:
            if self.jail_sentence_left > 0:
                self.jail_sentence_left -= 1
            else:
                self.public_state = PublicState.ORDINARY
            return

        observation = self._build_observation(sim)
        posts = sim.social_feed.sample_posts(self.post_context_size, self.sns_activity)

        if self.use_llm and self.step_counter % self.llm_decision_interval == 0:
            decision = self._call_llm(sim, observation, posts)
        else:
            decision = self._fallback_decision(observation, posts)

        self._apply_decision(decision)
        self._execute_movement(sim)
        self._maybe_post_to_feed(sim, decision)
        self.step_counter += 1
        self.last_decision = decision

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_observation(self, sim: "Simulation") -> dict:
        nearby = sim.get_agents_in_radius(self.pos, self.vision)
        state_counts: dict[str, int] = {s.value: 0 for s in PublicState}
        nearby_utterances: list[str] = []
        nearby_cops = 0

        for agent in nearby:
            if agent is self:
                continue
            if isinstance(agent, CitizenAgent):
                state_counts[agent.public_state.value] += 1
                if agent.utterance:
                    nearby_utterances.append(f"[{agent.public_state.value}] {agent.utterance}")
            elif isinstance(agent, CopAgent):
                nearby_cops += 1

        recent_utterances = nearby_utterances[-self.message_context_size:]
        crowd_density = len(nearby) / max(1, (2 * self.vision + 1) ** 2)
        landmark = sim.get_landmark_at(self.pos)

        return {
            "pos": self.pos,
            "landmark": landmark or "none",
            "nearby_by_state": state_counts,
            "nearby_cops": nearby_cops,
            "crowd_density": round(crowd_density, 2),
            "recent_utterances": recent_utterances,
        }

    # ------------------------------------------------------------------
    # LLM decision
    # ------------------------------------------------------------------

    def _call_llm(self, sim: "Simulation", observation: dict, posts: list[SocialPost]) -> dict:
        from prompts import CITIZEN_SYSTEM_PROMPT, build_decision_prompt
        prompt = build_decision_prompt(self, observation, posts, sim.steps)
        raw = self.llm_client.generate(
            prompt=prompt,
            system_prompt=CITIZEN_SYSTEM_PROMPT,
            force_json=True,
        )
        known_rumor_ids = set(self.belief_rumors) | set(sim.triggered_events)
        known_rumor_ids.update(p.rumor_id for p in posts if p.rumor_id)
        return self._parse_and_validate(raw, known_rumor_ids, has_rumor_context=bool(known_rumor_ids))

    def _parse_and_validate(
        self,
        raw: str,
        known_rumor_ids: set[str] | None = None,
        has_rumor_context: bool = True,
    ) -> dict:
        """Parse LLM JSON output and validate/sanitize all fields."""
        if not raw:
            decision = self._fallback_decision({})
            decision["_decision_source"] = "llm_empty_fallback"
            return decision
        try:
            # Strip <think>...</think> blocks (qwen3 thinking mode)
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            # Try to extract JSON if wrapped in extra text
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
            data = json.loads(cleaned)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Agent {self.id} JSON parse error: {e}")
            decision = self._fallback_decision({})
            decision["_decision_source"] = "llm_parse_fallback"
            return decision

        # Validate intent
        intent = data.get("intent", "observe")
        if intent not in VALID_INTENTS:
            intent = "observe"
        if not has_rumor_context and intent in (
            "verify_rumor",
            "share_uncertain_info",
            "share_correction",
            "calm_others",
            "join_peaceful_protest",
        ):
            intent = "observe"

        # Validate public_state
        ps_val = data.get("public_state", self.public_state.value)
        if ps_val not in VALID_PUBLIC_STATE_VALUES:
            ps_val = self.public_state.value

        # Validate emotion_update
        eu = data.get("emotion_update", {})
        emotion_update = {
            "anger": _clamp(eu.get("anger", self.anger)),
            "fear": _clamp(eu.get("fear", self.fear)),
            "solidarity": _clamp(eu.get("solidarity", self.solidarity)),
            "trust_official": _clamp(eu.get("trust_official", self.trust_official)),
            "trust_neighbors": _clamp(eu.get("trust_neighbors", self.trust_neighbors)),
        }

        # Belief updates are accepted only for rumors the agent could have
        # actually encountered. This prevents the model from inventing future
        # event ids before an event has fired.
        belief_updates = []
        allowed_rumors = known_rumor_ids or set()
        for bu in data.get("belief_updates", []):
            if not isinstance(bu, dict):
                continue
            rid = str(bu.get("rumor_id", ""))
            if not rid or rid not in allowed_rumors:
                continue
            try:
                new_strength = _clamp(float(bu.get("new_strength", self.belief_rumors.get(rid, 0.0))))
            except (TypeError, ValueError):
                continue
            belief_updates.append({
                "rumor_id": rid,
                "stance": str(bu.get("stance", "uncertain")),
                "new_strength": new_strength,
                "reason": str(bu.get("reason", "")),
            })

        # Sanitize utterance
        utterance = _clean_japanese_text(str(data.get("utterance", "")))
        if _contains_violence(utterance):
            utterance = ""
        if not has_rumor_context and _looks_like_unprompted_suspicion(utterance):
            utterance = _ordinary_small_talk(self)

        # SNS post
        sns_post_raw = data.get("sns_post", {})
        if isinstance(sns_post_raw, dict):
            sns_post = {
                "should_post": bool(sns_post_raw.get("should_post", False)),
                "content": _clean_japanese_text(str(sns_post_raw.get("content", ""))),
                "emotional_tone": str(sns_post_raw.get("emotional_tone", "calm")),
            }
        else:
            sns_post = {"should_post": False, "content": "", "emotional_tone": "calm"}
        if _contains_violence(sns_post["content"]):
            sns_post["content"] = ""
            sns_post["should_post"] = False
        if not has_rumor_context and _looks_like_unprompted_suspicion(sns_post["content"]):
            sns_post["content"] = ""
            sns_post["should_post"] = False

        return {
            "thinking": str(data.get("thinking", "")),
            "interpretation": str(data.get("interpretation", "")),
            "belief_updates": belief_updates,
            "emotion_update": emotion_update,
            "intent": intent,
            "target": INTENT_TO_LANDMARK.get(intent),
            "action_reason": str(data.get("action_reason", "")),
            "utterance": utterance,
            "sns_post": sns_post,
            "memory_to_add": str(data.get("memory_to_add", "")),
            "public_state": ps_val,
            "_decision_source": "llm",
        }

    # ------------------------------------------------------------------
    # Fallback (rule-based, no LLM)
    # ------------------------------------------------------------------

    def _fallback_decision(self, observation: dict, posts: list | None = None) -> dict:
        """Minimal rule-based fallback used when LLM is unavailable or fails."""
        nearby_cops = observation.get("nearby_cops", 0)
        state_counts = observation.get("nearby_by_state", {})
        participants_nearby = state_counts.get("participant", 0)
        observers_nearby = state_counts.get("observer", 0)

        # Propagate beliefs from SNS posts (what LLM would do in full mode)
        belief_updates = []
        if posts:
            for post in posts:
                rid = post.rumor_id
                if not rid:
                    continue
                ts = post.truth_status
                # Rumor-spreading posts increase belief; corrections decrease it
                if ts in ("FALSE", "MISLEADING", "UNVERIFIED"):
                    susceptibility = (1.0 - self.media_literacy * 0.6) * (1.0 - self.trust_official * 0.3)
                    delta = post.credibility_score * susceptibility * 0.35
                    new_strength = min(0.85, self.belief_rumors.get(rid, 0.0) + delta)
                elif ts == "CORRECTION":
                    # Weight by trust_official: higher trust = larger correction effect
                    delta = post.credibility_score * (0.2 + self.trust_official * 0.4)
                    new_strength = _clamp(self.belief_rumors.get(rid, 0.0) - delta)
                else:
                    continue
                if rid not in self.belief_rumors or abs(new_strength - self.belief_rumors.get(rid, 0.0)) > 0.01:
                    belief_updates.append({"rumor_id": rid, "new_strength": new_strength})
                    self.belief_rumors[rid] = new_strength  # apply immediately for this step's logic

        # Max belief strength across all known rumors
        max_belief = max(self.belief_rumors.values(), default=0.0)

        # Compute rough fear/anger nudge from environment
        fear_delta = nearby_cops * 0.06
        fear_decay = 0.06 if nearby_cops == 0 else 0.01
        # Anger rises above 0.3 belief threshold, weighted by hardship
        anger_stimulus = max(0.0, max_belief - 0.3) * self.hardship * 0.15
        anger_delta = anger_stimulus + participants_nearby * 0.04
        # Anger decays faster (toward baseline) — net decay when no stimulus
        anger_decay = 0.025
        solidarity_delta = participants_nearby * 0.04 + observers_nearby * 0.01

        new_anger = _clamp(self.anger + anger_delta - anger_decay)
        new_fear = _clamp(self.fear + fear_delta - fear_decay)
        new_solidarity = _clamp(self.solidarity + solidarity_delta - 0.005)

        # Determine intent by simple heuristics
        if self.public_state == PublicState.WITHDRAWN:
            if new_fear < 0.25 and nearby_cops == 0:
                # Recovered from withdrawal — reassess based on current beliefs
                if new_anger > 0.38 and max_belief > 0.35:
                    intent = "join_peaceful_protest"
                    ps = "participant"
                elif max_belief > 0.3:
                    intent = "observe"
                    ps = "observer"
                else:
                    intent = "stay_home"
                    ps = "ordinary"
            else:
                intent = "retreat"
                ps = "withdrawn"
        elif new_fear > 0.55 or nearby_cops > 2:
            intent = "retreat"
            ps = "withdrawn"
        elif new_anger > 0.38 and max_belief > 0.35 and nearby_cops < 2 and new_fear < 0.55:
            # High anger + belief drives participation (threshold tempered by fear/cops)
            intent = "join_peaceful_protest"
            ps = "participant"
        elif self.public_state == PublicState.PARTICIPANT and participants_nearby > 0 and nearby_cops < 3:
            intent = "join_peaceful_protest"
            ps = "participant"
        elif max_belief > 0.5 or observers_nearby + participants_nearby > 2:
            intent = "verify_rumor" if max_belief > 0.5 else "observe"
            ps = "observer"
        else:
            intent = "stay_home"
            ps = "ordinary"

        return {
            "thinking": "",
            "interpretation": "",
            "belief_updates": belief_updates,
            "emotion_update": {
                "anger": new_anger, "fear": new_fear, "solidarity": new_solidarity,
                "trust_official": self.trust_official, "trust_neighbors": self.trust_neighbors,
            },
            "intent": intent,
            "target": INTENT_TO_LANDMARK.get(intent),
            "action_reason": "fallback rule-based decision",
            "utterance": "",
            "sns_post": {"should_post": False, "content": "", "emotional_tone": "calm"},
            "memory_to_add": "",
            "public_state": ps,
            "_decision_source": "fallback",
        }

    # ------------------------------------------------------------------
    # Apply decision
    # ------------------------------------------------------------------

    def _apply_decision(self, decision: dict) -> None:
        eu = decision.get("emotion_update", {})
        # Keep emotional inertia so one cautious LLM response does not erase
        # accumulated anger/fear/solidarity in a single step.
        self.anger = _clamp(max(eu.get("anger", self.anger), self.anger * 0.70))
        self.fear = _clamp(max(eu.get("fear", self.fear), self.fear * 0.70))
        self.solidarity = _clamp(max(eu.get("solidarity", self.solidarity), self.solidarity * 0.85))
        self.trust_official = _clamp(eu.get("trust_official", self.trust_official))
        self.trust_neighbors = _clamp(eu.get("trust_neighbors", self.trust_neighbors))

        for bu in decision.get("belief_updates", []):
            rid = bu.get("rumor_id", "")
            strength = bu.get("new_strength")
            if rid and strength is not None:
                self.belief_rumors[rid] = _clamp(float(strength))

        ps_val = decision.get("public_state", self.public_state.value)
        try:
            self.public_state = PublicState(ps_val)
        except ValueError:
            pass

        self.current_intent = decision.get("intent", "stay_home")
        self.current_target = decision.get("target") or INTENT_TO_LANDMARK.get(self.current_intent)
        self.utterance = decision.get("utterance", "")
        self.private_stance = decision.get("interpretation", self.private_stance)

        mem = decision.get("memory_to_add", "")
        if mem:
            self.memory.append(mem)

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _execute_movement(self, sim: "Simulation") -> None:
        if self.current_intent == "stay_home":
            # Stay roughly in place (small drift only)
            new_pos = sim.random_adjacent(self.pos)
            sim.move_agent(self, new_pos)
            return
        target = self.current_target
        if not target or target not in sim.landmarks:
            new_pos = sim.random_adjacent(self.pos)
        else:
            center = sim.landmarks[target]["center"]
            new_pos = sim.step_toward(self.pos, tuple(center))
        sim.move_agent(self, new_pos)

    # ------------------------------------------------------------------
    # SNS posting
    # ------------------------------------------------------------------

    def _maybe_post_to_feed(self, sim: "Simulation", decision: dict) -> None:
        sns = decision.get("sns_post", {})
        if not sns.get("should_post"):
            return
        if random.random() > self.sns_activity:
            return
        content = sns.get("content", "")
        if not content:
            return

        # Infer truth_status from intent
        intent = decision.get("intent", "")
        if intent == "share_correction":
            truth_status = "CORRECTION"
        elif intent == "share_uncertain_info":
            truth_status = "UNVERIFIED"
        elif self.belief_rumors and max(self.belief_rumors.values(), default=0) > 0.6:
            truth_status = "UNVERIFIED"
        else:
            truth_status = "TRUE"

        post = SocialPost(
            post_id=str(uuid.uuid4())[:8],
            step=sim.steps,
            author_id=self.id,
            author_type="citizen",
            content=content,
            rumor_id=next(iter(self.belief_rumors), ""),
            truth_status=truth_status,
            emotional_tone=sns.get("emotional_tone", "calm"),
            reach=max(1, int(self.sns_activity * 20)),
            credibility_score=self.media_literacy * 0.5 + 0.25,
        )
        sim.social_feed.add_post(post)


# ---------------------------------------------------------------------------
# CopAgent (rule-based)
# ---------------------------------------------------------------------------

class CopAgent:
    """Rule-based safety officer. Moves toward crowds, detains participants."""

    def __init__(self, agent_id: int, pos: tuple[int, int], config: dict) -> None:
        self.id = agent_id
        self.pos = pos
        self.vision: int = config.get("agents", {}).get("vision", 3)
        self.max_jail_term: int = config.get("agents", {}).get("max_jail_term", 8)
        self.utterance: str = ""
        self.public_state: PublicState = PublicState.ORDINARY  # for DataCollector consistency

    def step(self, sim: "Simulation") -> None:
        self.utterance = ""
        nearby = sim.get_agents_in_radius(self.pos, self.vision)
        participants = [a for a in nearby if isinstance(a, CitizenAgent)
                        and a.public_state == PublicState.PARTICIPANT]

        # Arrest with low probability
        if participants and random.random() < 0.25:
            target_citizen = random.choice(participants)
            self._detain(target_citizen, sim)

        # Move toward the direction with most participants, or toward central_square
        self._move_toward_crowd(sim, nearby)

    def _detain(self, citizen: CitizenAgent, sim: "Simulation") -> None:
        citizen.public_state = PublicState.DETAINED
        citizen.jail_sentence_left = random.randint(3, self.max_jail_term)
        citizen.utterance = ""
        import uuid as _uuid
        post = SocialPost(
            post_id=str(_uuid.uuid4())[:8],
            step=sim.steps,
            author_id=self.id,
            author_type="cop_event",
            content=f"安全担当官が市民を一時拘束した（付近エリア）。",
            rumor_id="",
            truth_status="TRUE",
            emotional_tone="calm",
            reach=10,
            credibility_score=0.9,
        )
        sim.social_feed.add_post(post)
        logger.debug(f"Cop {self.id} detained Citizen {citizen.id} at step {sim.steps}")

    def _move_toward_crowd(self, sim: "Simulation", nearby: list) -> None:
        participants_nearby = [a for a in nearby
                               if isinstance(a, CitizenAgent)
                               and a.public_state in (PublicState.PARTICIPANT, PublicState.OBSERVER)]
        if participants_nearby:
            # Move toward centroid of participants
            avg_x = sum(a.pos[0] for a in participants_nearby) / len(participants_nearby)
            avg_y = sum(a.pos[1] for a in participants_nearby) / len(participants_nearby)
            new_pos = sim.step_toward(self.pos, (int(avg_x), int(avg_y)))
        else:
            # Patrol toward central_square
            center = sim.landmarks.get("central_square", {}).get("center", [10, 10])
            new_pos = sim.step_toward(self.pos, tuple(center))
        sim.move_agent(self, new_pos)


# ---------------------------------------------------------------------------
# InfluencerAgent
# ---------------------------------------------------------------------------

class InfluencerAgent:
    """
    SNS influencer that posts at regular intervals.
    Alignment determines the content bias.
    """

    ALIGNMENT_TRUTH_MAP = {
        "sensational":        ("MISLEADING", "angry"),
        "skeptical":          ("UNVERIFIED", "skeptical"),
        "official_friendly":  ("CORRECTION", "calm"),
        "protest_sympathetic": ("UNVERIFIED", "urgent"),
    }

    def __init__(
        self,
        agent_id: int,
        pos: tuple[int, int],
        alignment: str,
        reach_base: int,
        posting_interval: int,
    ) -> None:
        self.id = agent_id
        self.pos = pos
        self.alignment = alignment
        self.reach_base = reach_base
        self.posting_interval = posting_interval
        self.step_counter = 0
        self.utterance: str = ""
        self.public_state: PublicState = PublicState.ORDINARY

    def step(self, sim: "Simulation") -> None:
        self.utterance = ""
        if self.step_counter % self.posting_interval == 0 and sim.steps > 0:
            self._post(sim)
        self.step_counter += 1

    def _post(self, sim: "Simulation") -> None:
        import uuid as _uuid
        truth_status, emotional_tone = self.ALIGNMENT_TRUTH_MAP.get(
            self.alignment, ("UNVERIFIED", "calm")
        )
        rumor_id = ""
        content = ""

        if self.alignment == "sensational" and sim.triggered_events:
            evt_id = random.choice(list(sim.triggered_events))
            rumor = sim.event_rumors.get(evt_id, "")
            rumor_id = evt_id
            content = f"【拡散希望】{rumor} この情報は多くの人が確認しています。"
        elif self.alignment == "official_friendly" and sim.triggered_events:
            evt_id = random.choice(list(sim.triggered_events))
            truth = sim.event_truths.get(evt_id, "")
            rumor_id = evt_id
            content = f"落ち着いてください。実際には：{truth} 冷静な判断をお願いします。"
            truth_status = "CORRECTION"
        elif self.alignment == "skeptical":
            if sim.triggered_events:
                content = "流れている情報の多くは確認されていません。情報源を確認しましょう。"
            else:
                content = "駅前と配給所はいつも通り少し混んでいます。急ぐ人は時間に余裕を。"
                truth_status = "TRUE"
                emotional_tone = "calm"
        elif self.alignment == "protest_sympathetic" and sim.triggered_events:
            evt_id = random.choice(list(sim.triggered_events))
            rumor_id = evt_id
            content = "市民の声を聞いてください。何かが起きています。集まりましょう。"
        else:
            return

        post = SocialPost(
            post_id=str(_uuid.uuid4())[:8],
            step=sim.steps,
            author_id=self.id,
            author_type="influencer",
            content=content,
            rumor_id=rumor_id,
            truth_status=truth_status,
            emotional_tone=emotional_tone,
            reach=self.reach_base + random.randint(-5, 10),
            credibility_score=0.5,
        )
        sim.social_feed.add_post(post)


# ---------------------------------------------------------------------------
# OfficialAgent
# ---------------------------------------------------------------------------

class OfficialAgent:
    """Posts official corrections after a configurable delay."""

    def __init__(
        self,
        agent_id: int,
        response_delay: int,
        credibility: float = 0.8,
    ) -> None:
        self.id = agent_id
        self.response_delay = response_delay
        self.credibility = credibility
        self.corrections_posted: set[str] = set()
        self.utterance: str = ""
        self.public_state: PublicState = PublicState.ORDINARY
        self.pos = (0, 0)  # placeholder (no grid presence needed)

    def step(self, sim: "Simulation") -> None:
        self.utterance = ""
        if sim.steps < self.response_delay:
            return
        import uuid as _uuid
        for evt in sim.events:
            eid = evt["event_id"]
            if eid in self.corrections_posted:
                continue
            if eid not in sim.triggered_events:
                continue
            evt_fire_step = evt.get("step", 0)
            if evt_fire_step >= sim.steps:
                continue  # Don't correct same step it fires
            if evt_fire_step + 10 > sim.steps:
                continue  # Minimum 10-step lag per event
            truth = sim.event_truths.get(eid, "")
            if not truth:
                continue

            content = (
                f"【公式発表】{truth} "
                f"流れている不確かな情報には注意してください。"
            )
            post = SocialPost(
                post_id=str(_uuid.uuid4())[:8],
                step=sim.steps,
                author_id=self.id,
                author_type="official",
                content=content,
                rumor_id=eid,
                truth_status="CORRECTION",
                emotional_tone="calm",
                reach=40,
                credibility_score=self.credibility,
            )
            sim.social_feed.add_post(post)
            self.corrections_posted.add(eid)
            logger.info(f"Official posted correction for {eid} at step {sim.steps}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _contains_violence(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in VIOLENCE_KEYWORDS)


def _clean_japanese_text(text: str) -> str:
    replacements = {
        " heard ": " 聞いて ",
        "heard": "聞いて",
        " maybe ": " たぶん ",
        "maybe": "たぶん",
        " check ": " 確認 ",
        "check": "確認",
        " okay ": " 大丈夫 ",
        " ok ": " 大丈夫 ",
        "OK": "大丈夫",
        "SNS": "SNS",
        "LLM": "LLM",
        "小集会地点A": "近所の集まり場所",
        "小集会地点B": "近所の集まり場所",
        "小集会場所A": "近所の集まり場所",
        "小集会場所B": "近所の集まり場所",
        "場所A": "近所の集まり場所",
        "場所B": "近所の集まり場所",
        "central_square": "中央広場",
        "city_hall": "市庁舎",
        "media_zone": "メディア地区",
        "home_area": "住宅街",
        "exit_zone": "退避できる場所",
    }
    cleaned = text
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _looks_like_unprompted_suspicion(text: str) -> bool:
    return any(kw in text for kw in UNPROMPTED_SUSPICION_KEYWORDS)


def _ordinary_small_talk(agent: CitizenAgent) -> str:
    topic = getattr(agent, "daily_topic", "")
    if "駅前" in topic:
        return "駅前、今日は少し混んでるね。早めに用事を済ませよう。"
    if "配給" in topic or "店" in topic:
        return "買い物の列、今日は少し長いね。夕飯までに戻れるかな。"
    if "仕事" in topic:
        return "今日は少し疲れたな。帰ったら早めに休もう。"
    if "家族" in topic or "友人" in topic:
        return "あとで家族に連絡しておこう。今日は少し帰りが遅くなりそう。"
    if "天気" in topic:
        return "今日は少し肌寒いね。帰り道、気をつけよう。"
    if "市庁舎" in topic:
        return "市庁舎の手続き、今日のうちに確認しておきたいな。"
    if "メディア" in topic:
        return "地域ニュース、あとで確認しておこう。生活情報だけでも助かるから。"
    return "おはよう。今日はいつも通り、用事を済ませて帰ろう。"
