# llm-router

**Cost/latency-aware model routing with semantic caching.** I built this as a QA Lead moving into AI QA: when every prompt hits the same frontier model, you're overpaying for the easy 50% of traffic. This repo routes each task to the cheapest tier that can handle it, caches semantically similar prompts, and proves the savings with a deterministic simulation — including a check that routing never silently degrades quality.

No API keys, no network, fully deterministic. Everything runs offline on mock backends.

## Architecture

```
                        +------------------+
                        | data/workload    |
                        | .jsonl (40 tasks |
                        | complexity 1-10, |
                        | quality floors)  |
                        +--------+---------+
                                 |
                                 v
              +----------------------------------+
              |      run_simulation.py (CLI)     |
              |  routed (router+cache) vs naive  |
              |  (always-pro) baseline           |
              +-----+----------------------+-----+
                    |                      |
        +-----------v-----------+  +-------v-----------+
        | SemanticCache         |  | Router (router.py)|
        | (cache.py)            |  | policy:           |
        | TF-IDF + cosine,      |  | complexity        |
        | threshold 0.85        |  | threshold,        |
        | tracks hit rate,      |  | latency SLO,      |
        | cost/latency saved    |  | cost budgets      |
        +-----+-----------+-----+  +--------+----------+
              | hit       | miss          |
              |           v               v
              |   +-------+-------+  +----+---------+
              |   | backends.py   |  | metrics.py   |
              +-->| MockFlash /   |  | p95, mean,   |
                  | MockPro       |  | cost rollups |
                  | deterministic |  +------+-------+
                  | quality model |         |
                  +---------------+         v
                                  reports/routing_report.md
                                  (comparison table, ASCII
                                   bar charts, regressions)
```

Flow: the CLI loads the 40-task workload, runs it through the routed strategy (semantic cache → router → tier backend) and the naive always-pro baseline, aggregates cost/latency/quality with `metrics.py`, and writes `reports/routing_report.md`.

## The routing contract

The tier table lives in `src/config.py` and is the single source of truth:

| Tier | Price/1k tokens | Base latency | +/complexity pt | Capability | Max quality |
| --- | --- | --- | --- | --- | --- |
| `flash` | $0.00025 | 120 ms | 45 ms | 4.0 | 0.95 |
| `pro` | $0.00400 | 850 ms | 260 ms | 9.5 | 1.00 |

Quality is deterministic: full quality up to the tier's capability, then `capability / complexity` decay — so sending a complexity-8 task to `flash` scores 0.475 instead of 1.0, and the simulation can measure that consequence.

Routing decision order (`src/router.py`), every decision carrying an audit-trail `reasons` list:

1. **Complexity threshold** — complexity ≤ threshold → `flash`, else `pro` (at-threshold is inclusive → `flash`).
2. **Latency SLO** — if the pro estimate breaches the SLO, fall back to `flash` when flash fits.
3. **Budgets** — per-task cost cap or an exhausted session budget forces `flash`.

## The semantic cache

`src/cache.py` is embedding-free: TF-IDF vectors over the stored prompt corpus plus cosine similarity, stdlib only (`re`, `math`). A lookup at or above the similarity threshold (0.85, inclusive) is a hit and records the estimated cost/latency the backend call would have incurred as saved. Near-duplicate paraphrases hit; unrelated prompts miss.

## Quickstart

```bash
git clone <this-repo> && cd llm-router
pip install -r requirements.txt   # pytest only; src/ is stdlib-only

# Run the 40-task simulation (mock mode — no API key, no network)
python src/run_simulation.py

# Run the unit tests
pytest -q
```

### CLI options

```
python src/run_simulation.py [--workload data/workload.jsonl] [--out reports/routing_report.md]
                             [--complexity-threshold 4.0] [--cache-threshold 0.85]
```

The report lands in `reports/routing_report.md` (gitignored; `.gitkeep` holds the dir).

## Sample output

```
$ python src/run_simulation.py
Tasks: 40
Naive : cost=$0.0055 p95=3170ms regressions=0
Routed: cost=$0.0032 p95=3170ms regressions=0 cache_hit_rate=12.5%
Report: reports/routing_report.md
```

Excerpt from `reports/routing_report.md`:

| Metric | Naive (always-pro) | Routed + semantic cache | Delta |
| --- | --- | --- | --- |
| Total cost | $0.0055 | $0.0032 | 41.5% cheaper |
| Mean latency | 2073 ms | 1460 ms | 29.6% faster |
| p95 latency | 3170 ms | 3170 ms | 0.0% faster |
| Quality regressions | 0 | 0 | — |

```
naive   [############################] $0.0055
routed  [################------------] $0.0032
```

Routing mix: 14 tasks → flash, 21 → pro, 5 cache hits. The p95 is unchanged because the tail is dominated by hard tasks that correctly stay on pro — the router saves cost and latency on the other traffic without touching the tail. Zero quality regressions: every routed task met its quality floor.

## Production swap (honest note)

The mock backends are stand-ins, not benchmarks. To run this for real: implement the `ModelBackend` protocol in `src/backends.py` with one API client per tier (keys from `.env`, see `.env.example`), measure real per-call latency and cost, and replace the deterministic quality function with a real signal — human labels or an LLM judge against each task's `min_quality`. The router, cache, policy table, and simulation harness don't change; only the measurement source does.

## Layout

```
llm-router/
├── src/
│   ├── config.py         # tier price/latency/quality table + RoutingPolicy knobs
│   ├── backends.py       # ModelBackend protocol, MockFlash/MockPro, estimators
│   ├── router.py         # threshold + SLO + budget routing with audit reasons
│   ├── cache.py          # TF-IDF/cosine semantic cache (stdlib only)
│   ├── metrics.py        # percentile/p95/mean/cost aggregation
│   └── run_simulation.py # CLI: routed vs naive, writes reports/routing_report.md
├── data/
│   └── workload.jsonl    # 40 tasks: prompt, complexity, min_quality, category
├── tests/                # pytest: router edges, cache boundaries, metrics math, backends
├── reports/              # generated report (gitignored, has .gitkeep)
├── .env.example          # placeholders for the real-API swap
└── .github/workflows/ci.yml  # pytest + mock-mode simulation smoke test
```

## Roadmap

- Latency-aware cache admission: only cache responses from tiers where the hit actually saves meaningful latency.
- Complexity estimator: derive task complexity from the prompt (length, entities, reasoning cues) instead of labels.
- Multi-tier support (3+ tiers) with Pareto-frontier policy selection.
- Real-backend adapters behind the `ModelBackend` protocol, with the simulation able to replay in "measure" mode.
- Quality-gate flag on the CLI: fail the build when regressions exceed a threshold.

## License

MIT — see `LICENSE`.
