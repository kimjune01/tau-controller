# tau-controller

PID controller for τ — the publisher's relevance threshold in a [power-diagram ad auction](https://kimjune01.github.io/set-it-and-forget-it).

The publisher sets one number: what percentage of conversations should include a recommendation. The controller adjusts τ (a distance threshold in embedding space) to hit that target.

## How it works

- **Per-conversation, not per-turn.** A conversation is a series of turns. One recommendation per conversation, max.
- **PID feedback loop.** If the recommendation rate exceeds the target, τ tightens. If it's below, τ loosens. Integral and derivative terms handle drift and sudden changes.
- **Runs on the publisher's infrastructure.** No exchange dependency.

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
