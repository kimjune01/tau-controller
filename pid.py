"""
PID controller for tau — the publisher's relevance threshold.

The publisher sets a target: what percentage of conversations should include
a recommendation. The controller adjusts tau (a distance threshold in embedding
space) to hit that target.

Unit: per-conversation, not per-turn. A conversation is a series of turns.
One recommendation per conversation, max.
"""

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

    _integral: float = field(default=0.0, init=False)
    _prev_error: float = field(default=0.0, init=False)
    _prev_time: float = field(default=0.0, init=False)

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
        now = time.monotonic()
        dt = now - self._prev_time
        if dt <= 0:
            return self.tau

        # Error: positive means rate is too high, tau should tighten (decrease)
        error = observed_rate - self.target_rate

        self._integral += error * dt

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
    """Tracks whether a conversation has already received a recommendation."""

    _conversations: dict = field(default_factory=dict)

    def start(self, conversation_id: str):
        self._conversations[conversation_id] = False

    def has_recommendation(self, conversation_id: str) -> bool:
        return self._conversations.get(conversation_id, False)

    def mark_recommended(self, conversation_id: str):
        self._conversations[conversation_id] = True

    def end(self, conversation_id: str) -> bool:
        """End a conversation. Returns whether it received a recommendation."""
        had_rec = self._conversations.pop(conversation_id, False)
        return had_rec


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
    update_interval: int = 100  # recalculate tau every N completed conversations

    def should_recommend(self, conversation_id: str, best_ad_distance: float) -> bool:
        """
        Should this turn include a recommendation?

        Args:
            conversation_id: unique ID for the conversation
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
        self._recent_total += 1
        if had_rec:
            self._recent_recommended += 1

        if self._recent_total >= self.update_interval:
            observed_rate = self._recent_recommended / self._recent_total
            self.controller.update(observed_rate)
            self._recent_total = 0
            self._recent_recommended = 0
