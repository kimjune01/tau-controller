import time
from unittest.mock import patch

from pid import TauController, ConversationTracker, RecommendationGate


def test_tau_decreases_when_rate_too_high():
    """If observed rate exceeds target, tau should tighten (decrease)."""
    ctrl = TauController(target_rate=0.10, tau=1.0)
    initial_tau = ctrl.tau

    with patch("pid.time") as mock_time:
        mock_time.monotonic.return_value = 1.0
        ctrl._prev_time = 0.0
        ctrl.update(observed_rate=0.20)  # double the target

    assert ctrl.tau < initial_tau


def test_tau_increases_when_rate_too_low():
    """If observed rate is below target, tau should loosen (increase)."""
    ctrl = TauController(target_rate=0.10, tau=1.0)
    initial_tau = ctrl.tau

    with patch("pid.time") as mock_time:
        mock_time.monotonic.return_value = 1.0
        ctrl._prev_time = 0.0
        ctrl.update(observed_rate=0.02)  # well below target

    assert ctrl.tau > initial_tau


def test_tau_stays_in_bounds():
    ctrl = TauController(target_rate=0.10, tau=0.02, tau_min=0.01, tau_max=10.0)

    with patch("pid.time") as mock_time:
        mock_time.monotonic.return_value = 1.0
        ctrl._prev_time = 0.0
        # massive overshoot
        ctrl.update(observed_rate=0.99)

    assert ctrl.tau >= ctrl.tau_min
    assert ctrl.tau <= ctrl.tau_max


def test_conversation_tracker_one_recommendation():
    tracker = ConversationTracker()
    tracker.start("conv-1")

    assert not tracker.has_recommendation("conv-1")
    tracker.mark_recommended("conv-1")
    assert tracker.has_recommendation("conv-1")

    had_rec = tracker.end("conv-1")
    assert had_rec is True


def test_conversation_tracker_no_recommendation():
    tracker = ConversationTracker()
    tracker.start("conv-2")
    had_rec = tracker.end("conv-2")
    assert had_rec is False


def test_gate_recommends_once_per_conversation():
    ctrl = TauController(target_rate=0.10, tau=1.0)
    gate = RecommendationGate(controller=ctrl)

    gate.tracker.start("conv-1")

    # First turn with a close ad: should recommend
    assert gate.should_recommend("conv-1", best_ad_distance=0.5) is True
    # Second turn: already recommended, should not
    assert gate.should_recommend("conv-1", best_ad_distance=0.3) is False


def test_gate_rejects_distant_ads():
    ctrl = TauController(target_rate=0.10, tau=1.0)
    gate = RecommendationGate(controller=ctrl)

    gate.tracker.start("conv-1")

    # Ad too far away
    assert gate.should_recommend("conv-1", best_ad_distance=2.0) is False


def test_gate_updates_tau_after_interval():
    ctrl = TauController(target_rate=0.10, tau=1.0)
    gate = RecommendationGate(controller=ctrl, update_interval=10)

    initial_tau = ctrl.tau

    with patch("pid.time") as mock_time:
        mock_time.monotonic.return_value = 1.0
        ctrl._prev_time = 0.0

        # Simulate 10 conversations, all getting recommendations (100% rate)
        for i in range(10):
            cid = f"conv-{i}"
            gate.tracker.start(cid)
            gate.should_recommend(cid, best_ad_distance=0.1)  # always close
            gate.on_conversation_end(cid)

    # 100% recommendation rate vs 10% target — tau should have tightened
    assert ctrl.tau < initial_tau


def test_convergence():
    """Tau should converge toward a value that produces the target rate."""
    # Uses default gains (kp=0.5, ki=0.05, kd=0.1)
    ctrl = TauController(target_rate=0.10, tau=1.5)
    gate = RecommendationGate(controller=ctrl, update_interval=100)

    import random
    random.seed(42)

    with patch("pid.time") as mock_time:
        t = 0.0
        ctrl._prev_time = t

        for i in range(5000):
            t += 0.01
            mock_time.monotonic.return_value = t

            cid = f"conv-{i}"
            gate.tracker.start(cid)
            distance = random.uniform(0.0, 3.0)
            gate.should_recommend(cid, best_ad_distance=distance)
            gate.on_conversation_end(cid)

    # With uniform distances in [0, 3], rate = tau/3.
    # Target is 0.10, so equilibrium tau ≈ 0.30.
    # Should converge within 50% of equilibrium
    equilibrium_tau = 0.10 * 3.0
    assert abs(ctrl.tau - equilibrium_tau) / equilibrium_tau < 0.50
