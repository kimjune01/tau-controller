"""
PID controller for tau — the publisher's relevance threshold.

The publisher sets a target: what percentage of conversations should include
a recommendation. The controller adjusts tau (a distance threshold in embedding
space) to hit that target.

Unit: per-conversation, not per-turn. A conversation is a series of turns.
One recommendation per conversation, max.

PRIVACY: This code never sees embeddings, user data, or content. It only
processes distances (floats) and opaque conversation IDs (strings).
Conversation IDs MUST be opaque tokens (e.g. UUIDs), never PII or PHI.
"""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class TauController:
    """PID controller that adjusts tau to hit a target recommendation rate."""

    target_rate: float  # e.g. 0.10 for 10% of conversations
    kp: float = 0.5    # proportional gain
    ki: float = 0.05   # integral gain
    kd: float = 0.1    # derivative gain
    tau: float = 1.0    # initial distance threshold
    tau_min: float = 0.01
    tau_max: float = 10.0
    integral_max: float = 10.0  # anti-windup clamp

    _integral: float = field(default=0.0, init=False)
    _prev_error: float = field(default=0.0, init=False)
    _prev_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        self._prev_time = time.monotonic()

    def update(self, observed_rate: float) -> float:
        """
        Update tau based on the observed recommendation rate.

        Args:
            observed_rate: fraction of recent conversations that included
                           a recommendation (e.g. 0.14 for 14%)

        Returns:
            The new tau value.
        """
        with self._lock:
            now = time.monotonic()
            dt = now - self._prev_time
            if dt <= 0:
                return self.tau

            # Error: positive means rate is too high, tau should tighten (decrease)
            error = observed_rate - self.target_rate

            self._integral += error * dt
            self._integral = max(-self.integral_max, min(self.integral_max, self._integral))

            derivative = (error - self._prev_error) / dt

            adjustment = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)

            # Negative adjustment = tighten tau (lower distance threshold)
            self.tau -= adjustment
            self.tau = max(self.tau_min, min(self.tau_max, self.tau))

            self._prev_error = error
            self._prev_time = now

            return self.tau


@dataclass
class ConversationTracker:
    """
    Tracks whether a conversation has already received a recommendation.

    Conversation IDs MUST be opaque tokens (UUIDs), never PII or PHI.
    Conversations that exceed ttl_seconds are automatically evicted.
    """

    ttl_seconds: float = 3600.0  # evict conversations older than 1 hour
    max_conversations: int = 100_000  # hard cap to prevent memory exhaustion

    _conversations: dict = field(default_factory=dict, repr=False)
    _start_times: dict = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self, conversation_id: str):
        with self._lock:
            self._evict_expired()
            if len(self._conversations) >= self.max_conversations:
                self._evict_oldest()
            self._conversations[conversation_id] = False
            self._start_times[conversation_id] = time.monotonic()

    def has_recommendation(self, conversation_id: str) -> bool:
        with self._lock:
            return self._conversations.get(conversation_id, False)

    def mark_recommended(self, conversation_id: str):
        with self._lock:
            if conversation_id in self._conversations:
                self._conversations[conversation_id] = True

    def end(self, conversation_id: str) -> bool:
        """End a conversation. Returns whether it received a recommendation."""
        with self._lock:
            had_rec = self._conversations.pop(conversation_id, False)
            self._start_times.pop(conversation_id, None)
            return had_rec

    def _evict_expired(self):
        """Remove conversations that exceeded the TTL. Caller must hold _lock."""
        now = time.monotonic()
        expired = [
            cid for cid, start in self._start_times.items()
            if now - start > self.ttl_seconds
        ]
        for cid in expired:
            self._conversations.pop(cid, None)
            self._start_times.pop(cid, None)

    def _evict_oldest(self):
        """Remove the oldest conversation. Caller must hold _lock."""
        if not self._start_times:
            return
        oldest = min(self._start_times, key=self._start_times.get)
        self._conversations.pop(oldest, None)
        self._start_times.pop(oldest, None)


@dataclass
class RecommendationGate:
    """
    Decides whether a conversation should receive a recommendation.

    Combines the tau controller with conversation tracking. Call `should_recommend`
    on each turn. It returns True at most once per conversation, and only if the
    best available ad's distance is below the current tau.
    """

    controller: TauController
    tracker: ConversationTracker = field(default_factory=ConversationTracker)
    _recent_total: int = field(default=0, init=False)
    _recent_recommended: int = field(default=0, init=False)
    _counter_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    update_interval: int = 100  # recalculate tau every N completed conversations

    def should_recommend(self, conversation_id: str, best_ad_distance: float) -> bool:
        """
        Should this turn include a recommendation?

        Args:
            conversation_id: opaque token (UUID). MUST NOT be PII or PHI.
            best_ad_distance: distance from the query embedding to the
                              nearest ad's center in embedding space

        Returns:
            True if the ad should be shown. False if tau filters it out
            or the conversation already has a recommendation.
        """
        if self.tracker.has_recommendation(conversation_id):
            return False

        if best_ad_distance <= self.controller.tau:
            self.tracker.mark_recommended(conversation_id)
            return True

        return False

    def on_conversation_end(self, conversation_id: str):
        """Call when a conversation ends. Updates the observed rate."""
        had_rec = self.tracker.end(conversation_id)

        with self._counter_lock:
            self._recent_total += 1
            if had_rec:
                self._recent_recommended += 1

            if self._recent_total >= self.update_interval:
                observed_rate = self._recent_recommended / self._recent_total
                self.controller.update(observed_rate)
                self._recent_total = 0
                self._recent_recommended = 0
