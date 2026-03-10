"""
Simulate the tau PID controller across different target rates and conditions.
Generates convergence plots and summary statistics as evidence.
"""

import random
from unittest.mock import patch

import matplotlib.pyplot as plt

from pid import TauController, RecommendationGate


def simulate_steady_state(target_rate: float, n_conversations: int = 3000,
                          update_interval: int = 50, seed: int = 42,
                          ad_distance_range: float = 3.0) -> dict:
    """
    Simulate conversations and track tau convergence.

    Each conversation has a topic. The best available ad has a fixed distance
    to that topic (doesn't change turn to turn). The gate checks once per
    conversation whether that distance is below tau.
    """
    random.seed(seed)

    ctrl = TauController(target_rate=target_rate, tau=1.5, kp=0.5, ki=0.05, kd=0.1)
    gate = RecommendationGate(controller=ctrl, update_interval=update_interval)

    tau_history = []
    rate_chunks = []
    chunk_recs = 0
    chunk_total = 0

    with patch("pid.time") as mock_time:
        t = 0.0
        ctrl._prev_time = t

        for i in range(n_conversations):
            t += 0.01
            mock_time.monotonic.return_value = t

            cid = f"conv-{i}"
            gate.tracker.start(cid)

            # One best-ad distance per conversation (the topic determines it)
            best_distance = random.uniform(0.0, ad_distance_range)

            # Multiple turns, but the best ad is the same each turn
            n_turns = random.randint(1, 15)
            recommended = False
            for _ in range(n_turns):
                if gate.should_recommend(cid, best_ad_distance=best_distance):
                    recommended = True

            gate.on_conversation_end(cid)
            tau_history.append(ctrl.tau)

            chunk_total += 1
            if recommended:
                chunk_recs += 1
            if chunk_total == update_interval:
                rate_chunks.append(chunk_recs / chunk_total)
                chunk_recs = 0
                chunk_total = 0

    return {
        "target_rate": target_rate,
        "tau_history": tau_history,
        "rate_chunks": rate_chunks,
        "final_tau": ctrl.tau,
        "final_rate": rate_chunks[-1] if rate_chunks else 0,
        "update_interval": update_interval,
        # Theoretical: with uniform distances in [0, range], rate = tau / range
        # So equilibrium tau = target_rate * range
        "equilibrium_tau": target_rate * ad_distance_range,
    }


def simulate_shock(target_rate: float = 0.10, n_conversations: int = 4000,
                   shock_at: int = 2000, seed: int = 42) -> dict:
    """
    Simulate a sudden change: a new advertiser category enters the auction,
    making closer ads available. The distance distribution shifts.
    """
    random.seed(seed)

    ctrl = TauController(target_rate=target_rate, tau=1.5, kp=0.5, ki=0.05, kd=0.1)
    gate = RecommendationGate(controller=ctrl, update_interval=50)

    tau_history = []
    rate_chunks = []
    chunk_recs = 0
    chunk_total = 0

    with patch("pid.time") as mock_time:
        t = 0.0
        ctrl._prev_time = t

        for i in range(n_conversations):
            t += 0.01
            mock_time.monotonic.return_value = t

            cid = f"conv-{i}"
            gate.tracker.start(cid)

            # After shock, ads are closer (new category with tight targeting)
            if i >= shock_at:
                best_distance = random.uniform(0.0, 1.5)
            else:
                best_distance = random.uniform(0.0, 3.0)

            n_turns = random.randint(1, 15)
            recommended = False
            for _ in range(n_turns):
                if gate.should_recommend(cid, best_ad_distance=best_distance):
                    recommended = True

            gate.on_conversation_end(cid)
            tau_history.append(ctrl.tau)

            chunk_total += 1
            if recommended:
                chunk_recs += 1
            if chunk_total == 50:
                rate_chunks.append(chunk_recs / chunk_total)
                chunk_recs = 0
                chunk_total = 0

    return {
        "tau_history": tau_history,
        "rate_chunks": rate_chunks,
        "shock_at": shock_at,
        "target_rate": target_rate,
    }


def plot_convergence(results: list[dict], filename: str = "convergence.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for r in results:
        label = f"target={r['target_rate']:.0%}"
        axes[0].plot(r["tau_history"], label=label, alpha=0.8)
        axes[0].axhline(y=r["equilibrium_tau"], color="gray", linestyle=":", alpha=0.3)

    axes[0].set_ylabel("τ (distance threshold)")
    axes[0].set_title("Tau Convergence by Target Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for r in results:
        label = f"target={r['target_rate']:.0%}"
        x = [i * r["update_interval"] for i in range(len(r["rate_chunks"]))]
        axes[1].plot(x, r["rate_chunks"], label=label, alpha=0.8)
        axes[1].axhline(y=r["target_rate"], color="gray", linestyle="--", alpha=0.3)

    axes[1].set_ylabel("Recommendation Rate")
    axes[1].set_xlabel("Conversations")
    axes[1].set_title("Observed Recommendation Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_shock(result: dict, filename: str = "shock_recovery.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(result["tau_history"], color="steelblue", alpha=0.8)
    axes[0].axvline(x=result["shock_at"], color="red", linestyle="--", alpha=0.5, label="New category enters")
    axes[0].set_ylabel("τ (distance threshold)")
    axes[0].set_title("Tau Recovery After Shock (New Advertiser Category)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x = [i * 50 for i in range(len(result["rate_chunks"]))]
    axes[1].plot(x, result["rate_chunks"], color="steelblue", alpha=0.8)
    axes[1].axhline(y=result["target_rate"], color="gray", linestyle="--", alpha=0.5, label="Target")
    axes[1].axvline(x=result["shock_at"], color="red", linestyle="--", alpha=0.5, label="New category enters")
    axes[1].set_ylabel("Recommendation Rate")
    axes[1].set_xlabel("Conversations")
    axes[1].set_title("Recommendation Rate Recovery")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def print_summary(results: list[dict]):
    print("\n=== Convergence Summary ===\n")
    print(f"{'Target':>10} {'Equil. τ':>10} {'Final τ':>10} {'Final Rate':>12} {'|Error|':>10}")
    print("-" * 56)
    for r in results:
        error = abs(r["final_rate"] - r["target_rate"])
        print(f"{r['target_rate']:>10.0%} {r['equilibrium_tau']:>10.3f} "
              f"{r['final_tau']:>10.3f} {r['final_rate']:>12.1%} {error:>10.3f}")


if __name__ == "__main__":
    targets = [0.05, 0.10, 0.20, 0.30]
    results = [simulate_steady_state(t, n_conversations=5000, update_interval=100) for t in targets]

    print_summary(results)
    plot_convergence(results)

    shock = simulate_shock()
    plot_shock(shock)

    print("\nDone.")
