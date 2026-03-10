# tau-controller

PID controller for τ — the publisher's relevance threshold in a [power-diagram ad auction](https://kimjune01.github.io/set-it-and-forget-it).

The publisher sets one number: what percentage of conversations should include a recommendation. The controller adjusts τ (a distance threshold in embedding space) to hit that target.

## How it works

- **Per-conversation, not per-turn.** A conversation is a series of turns. One recommendation per conversation, max.
- **PID feedback loop.** If the recommendation rate exceeds the target, τ tightens. If it's below, τ loosens. Integral and derivative terms handle drift and sudden changes.
- **Runs on the publisher's infrastructure.** No exchange dependency.

## Simulation results

`uv run --with matplotlib simulate.py` runs 5,000 conversations across four target rates and a shock scenario.

### Convergence

The controller converges to the target recommendation rate from an arbitrary starting τ:

| Target | Equilibrium τ | Final τ | Final Rate | Error |
|--------|--------------|---------|------------|-------|
| 5% | 0.150 | 0.105 | 7% | 0.020 |
| 10% | 0.300 | 0.319 | 12% | 0.020 |
| 20% | 0.600 | 0.624 | 20% | 0.000 |
| 30% | 0.900 | 0.910 | 26% | 0.040 |

![Tau convergence by target rate](convergence.png)

### Shock recovery

When a new advertiser category enters (ad distances suddenly halve), τ tightens to maintain the target rate:

![Tau recovery after shock](shock_recovery.png)

## Usage

```python
from pid import TauController, RecommendationGate

controller = TauController(target_rate=0.10)  # 10% of conversations
gate = RecommendationGate(controller=controller)

# On each conversation start
gate.tracker.start(conversation_id)

# On each turn, check whether to show a recommendation
if gate.should_recommend(conversation_id, best_ad_distance):
    show_recommendation()

# On conversation end
gate.on_conversation_end(conversation_id)
```

## Tests

```
uv run --with pytest pytest test_pid.py -v
```

## License

MIT
