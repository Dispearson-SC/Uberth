# Decision log

Every entry records what was decided, why, and — where it applies — the
measurement that forced it. Entries are append-only: a decision that was later
reversed keeps its original entry with the reversal noted, because how we got
somewhere wrong is worth as much as where we ended up.

For the agent's architecture see `AGENT_MODEL.md`. For per-figure provenance
see the git history: every number below is recorded in the commit that
produced it.

---

## D1 — Deterministic, seeded, replayable simulator

**Decided:** the exogenous world is precomputed from a seed and frozen into a
serialisable `Scenario`; endogenous state evolves per tick.

**Why:** the brief asks for two agents running the same shift side by side. If
the world is random at run time, the comparison proves nothing. Freezing it
also lets a scenario be hand-authored — a street closure at exactly minute
143, when the courier is mid-route — rather than hoping the RNG cooperates.

**Consequence:** one independent RNG stream per concern via
`SeedSequence.spawn`. A single global RNG is the classic mistake: adding one
event type would reshuffle the order stream and destroy comparability.

---

## D2 — Scope cuts, made at hour zero

**Cut:** SUMO, OSRM in Docker, Redis, reinforcement learning, Kaggle order
datasets, multi-app operation, full VRPTW, an LLM inside the decision loop.

**Why each:** SUMO simulates individual vehicles when we need a per-zone time
multiplier — a function, not a simulator. OSRM costs hours of Docker for what
a cached graph gives. Redis for state that fits in a dict. RL cannot be
trained or defended in the time available, and it destroys explainability,
which is scored. Kaggle food-delivery sets are Indian — useful for
distribution shape, not geography.

**The LLM one is worth stating separately:** the `DecisionTrace` already
produces plain-language sentences. Putting a model in the critical path adds a
way for the demo to hang on stage and adds nothing.

---

## D3 — Restrict the operating area

**Decided:** Monterrey, San Pedro Garza García, San Nicolás de los Garza,
Guadalupe. About 90 H3 cells at resolution 7; 127 after buffering.

**Why:** a 127×127 matrix is seconds of Dijkstra instead of minutes. And it is
*more* realistic, not less — no real courier works an entire metropolitan
area. Restricting the area is fidelity, not a shortcut.

---

## D4 — Sublinear weighting of restaurants

**Measured:** DENUE `per_ocu` is heavily skewed — 83% of venues are "0 a 5
personas". Weighting linearly by staff lets roughly 130 large venues dominate
and the demand field collapses into a handful of points.

**Decided:** `sqrt(midpoint(per_ocu))` times a per-SCIAN delivery factor.
Verified: the 130 heaviest venues hold 4.4% of total weight.

---

## D5 — Traffic: the assumption that was falsified

**Originally decided:** borrow the daily congestion shape from TomTom's
measured Mexico City series, arguing that peak TIMING transfers between two
Mexican metros sharing work and meal schedules, while magnitude does not.

**Then measured, once a Traffic Stats trial covered our own polygon:**

```
hour    CDMX %    Monterrey multiplier
03:00     0.2       1.000  (reference, empirically the fastest hour)
08:00    84.9       1.451
11:00    55.5 <-    1.655     CDMX dips at midday. Monterrey does not.
12:00    57.0       1.679 <-  Monterrey's actual peak
18:00    89.3 <-    1.656     CDMX's peak is a plateau point here
23:00    13.6       1.395
```

**Outcome:** the assumption was WRONG on timing and right on magnitude. CDMX
is a sharp bimodal commute city; Monterrey is one broad plateau from 10:00 to
20:00 peaking at NOON. Had we shipped the borrowed curve, the simulator would
have taught the agent a midday valley that does not exist here — a wrong
routing bias through half of every day shift.

**Kept:** `MONTERREY_CONGESTION_SCALE` was retired entirely. Both shape and
magnitude are now measured. Weekday is 100% measured Monterrey; the trial
capped us at 24 time sets so weekend remains the real weekday curve reshaped
by a CDMX weekend ratio, capped asymmetrically — the raw ratio reaches 7.7x at
02:00 where CDMX's own baseline is near zero, and transferring that put
weekend 01:00 above Monterrey's measured midday peak.

**Lesson:** a reasonable assumption is still an assumption. It was worth three
failed download attempts and a fight with an async API to find out.

---

## D6 — Surge is a mechanism, not a multiplier

**Decided:** `surge = clip(1 + k·(demand/supply − equilibrium))`, with a
competing-courier supply field that migrates toward high surge **with a lag**.

**Why:** most simulators fake surge with `payout × random(1.2, 2.0)`. The lag
is the whole point: surge spikes, couriers pour in, by the time they arrive the
imbalance has closed, surge collapses. **The in-app heatmap therefore lies
without being scripted to** — because everyone sees the same signal and nobody
teleports.

**Proof it is emergent:** forcing the lag to zero collapses temporal standard
deviation from 0.65 to 0.02. That single number is the answer to "did you
program the oscillation?"

**Two defects found while calibrating, both worth recording:**
- `min_supply_reserve_fraction` was computed every minute and never consulted.
  The fix was written in prose in a comment and never wired into the code.
  Without that floor, a sustained one-sided gap drained a cell toward zero and
  exploded `demand/eps`.
- The migration gain sat past the stability threshold of the delayed feedback
  loop. Sweeping it alone flips between flatlined and wild between 0.048 and
  0.050 — a textbook bifurcation.

**Method note, the most transferable thing here:** what separated three
competing hypotheses was splitting variance into TEMPORAL (within a cell over
time) versus SPATIAL (across cells at one minute). Spatial near zero with
temporal large means the cells are oscillating in lockstep, which is a
dynamics bug that no amount of tuning `k` can reach.

**Metric error I made:** I asked an agent to raise the standard deviation of
per-cell shift-long MEANS. That averages out by construction — it reads 0.079
no matter what. The correct measure is the spread ACROSS cells at the same
instant, which reads 0.244. The agent sacrificed good configurations chasing a
metric I defined wrong.

---

## D6b — Courier supply varies by hour

**Decided:** supply follows its own hourly curve, deliberately falling faster
than demand at both edges of the day.

**Why:** supply was flat, so at 06:00 the model divided low demand by a full
fleet and produced a dead market with low earnings. Reality is the opposite,
and this came from the user, who drives: early shifts pay better precisely
because few couriers will work them while breakfast orders still exist. **A
simulator that contradicts what a courier knows from experience discredits
every other number it prints.**

**Then measured:** the early-shift advantage does NOT come from surge. Forcing
it there is arithmetically impossible inside the realism band — an agent spent
two hours proving that. It comes from THROUGHPUT: average congestion is 1.504
early against 1.629 at dinner, so travel is faster and more deliveries fit in
an hour. Measured that way the gap is +7.9%.

**Lesson:** when something will not come out after a lot of turning, the
question is whether it was ever the right knob.

---

## D7 — Hexagonal for what is new; leave the calibrated world alone

**Asked:** restructure everything to hexagonal architecture, add strict TDD.

**Decided:** hexagonal for the 60% not yet written, which is free. The
calibrated world stays behind the port it already had.

**Why not rewrite:** those 3,728 lines are not code, they are a **calibrated
instrument**. The surge sits a hair from its bifurcation, and the calibration
depends on the ORDER in which random numbers are drawn. Moving that code
changes the order, which changes the output, which means paying for the
calibration again — measured at over three hours of agent time. Refactoring
broken code is cheap; refactoring a calibrated instrument is not.

**On TDD:** strict TDD everywhere would have halved throughput with eight
hours left. Zero tests was genuinely dangerous, so the answer was
**characterization tests** — lock the already-verified numbers, in 30 minutes.
Plus test-first on the agent's decision logic specifically, which is pure
logic with no I/O and where correctness matters most.

**The sentinel:** a sha256 over all 27,073 serialised orders. Any behavioural
drift anywhere in the generation path moves it. Regenerate ONLY when a
calibration deliberately changed, and say in the commit which change moved it.

---

## D8 — Parallel agents, but the contract is never delegated

**Decided:** fan out to independent writers, but the orchestrator writes the
coordination contract by hand.

**Why:** with N agents against an undefined interface you get N interfaces.
Both times this worked, it worked because the contract was frozen first —
`world/timeline.py` for the exogenous producers, `core/ports.py` for
everything else.

**Evidence it paid:** `eval/` was written against the ports while `engine/`,
`platform/` and `agent/` did not yet exist, and surfaced four gaps in the
contract before anyone had written a line against a wrong assumption.

**On workflow orchestration:** rejected. The bottleneck is contract definition
and reviewing output against measured numbers — judgement that cannot be
scripted. Plain parallel calls give the same parallelism without the
machinery.

---

## D9 — Offers are a flow, not a menu

**Decided:** one offer at a time, fresh only, never stored, never re-offered.
A fresh offer CAN arrive while the courier is busy and be accepted, queueing
up to two jobs, strictly sequential.

**Why, and this came from the user:** an order that appeared ten minutes ago
has already been taken by someone else. Letting a busy courier reach back into
a backlog is fishing in an empty pond and would inflate every result. But
queueing a FRESH offer is real — it is how couriers chain deliveries without
idle gaps, and it is what Uber Eats and DiDi actually do.

It also matches the brief's own words: *"La app muestra un flujo de pedidos, y
el repartidor tiene segundos para aceptar o rechazar."* A flow, and seconds.

**Consequence:** the decision stops being "pick the best of five" and becomes
sequential take-it-or-leave-it against a reservation price — *is this good
enough, knowing another arrives in about eight minutes and rejecting costs me
idle time?* That is optimal stopping, and it is a richer problem than menu
selection. It also gives the policy a second lever: **when to commit**.
Accepting early kills idle time but commits blind; staying free keeps the
option and risks an empty screen.

---

## D10 — The agent pulls; it does not receive

**Reversal of the original design.** See `AGENT_MODEL.md` for the full
argument. In short: the engine pushing an `Observation` meant the engine was
deciding what the agent needed to know, and it meant the agent could not start
in a city where no engine exists. Our world is Monterrey by construction.

**The line:** the simulator stands in for reality and may use Mexico-only
sources. The agent exports to any city and may use only what a lat/lon and an
app screen yield. **DENUE is Mexico. OpenStreetMap is the planet.**

**Stated honestly:** this buys defensibility and portability, not performance.
If the agent queries everything it used to be handed, the decisions are
identical.

---

## D11 — No external traffic feed. OSM plus learning.

**Decided:** the agent's travel model is two layers and nothing else — an
OSM-derived free-flow matrix it builds itself once, corrected by a table
learned from its own completed trips. No vendor feed, and no pluggable slot
for one.

**Why, and the reason is not budget:** an external feed measures CAR probe
data. A courier on a motorcycle filters between lanes, takes gaps a car
cannot, and parks in thirty seconds. Their travel times are systematically
different from anything a car-derived feed reports. **The agent's own
completed trips are not a portable-but-inferior substitute — they are the
correct measurement**, taken on the actual vehicle over the actual routes.

**Secondary benefits:** one fewer dependency, one fewer credential, one fewer
thing that can fail on stage, and nothing in the agent a courier in any city
could not obtain for free.

---

## D12 — The demo plays a recording

**Decided:** shifts run headless, every tick is recorded to JSON, and the
dashboard is a dumb player over static data. One self-contained HTML file, no
build step, no server.

**Why:** nothing can stall on stage because nothing computes there. And when a
judge says "go back to the minute it rejected that order", scrubbing is
instant instead of a re-simulation. An 8-hour shift already computes in 3.3
seconds, so live computation buys nothing and risks everything.

**Also:** the km counter is as prominent as the pesos counter, because the
headline result is that the agent earns nearly the same money on 42% less
driving, and that does not land if kilometres are a footnote.

---

## Open items — diagnosed, not hidden

| Item | Status |
|---|---|
| `self_check` fails on deliveries/hour and MXN/h | Thresholds untouched. Dominant cycle component is the ride TO the restaurant, not kitchen wait. |
| Night window loses 22% to accept-everything | Cause found: the reservation price uses a lifetime-average arrival rate, so it stays at dinner-peak height through three dead hours after midnight. The obvious fix — a recency window — measured WORSE, because the same rate also inflates every destination's dead-minute estimate. Needs the two uses decoupled. |
| Day window surge at 25.8%, above the 6-18% band | A flat-profile control run shows 17.8%, so it predates the supply profile and lives in the surge constants. |
| Smart does not beat a 55-peso payout floor on MXN/h | −2.8% on the reference window, winning 3 of 6 seeds. It does beat accept-everything by +23.1%, 6 of 6. The honest claim is the kilometres, not the pesos. |
