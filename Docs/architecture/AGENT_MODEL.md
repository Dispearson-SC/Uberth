# The Agent Model

**Status:** design of record. Supersedes the push-based observation flow.
**Audience:** whoever builds, reviews or judges this system.

---

## 1. The one sentence

The platform shows a courier six fields. The agent computes a seventh the
platform will never show — **expected net MXN per hour** — by running its own
analysis over sources anyone could obtain, and it must do so in **any city**,
not only the one we happened to calibrate a simulator for.

---

## 2. Why the previous design was wrong

The engine used to build an `Observation` and hand it to the policy. That is
indefensible on three counts, and the third is fatal.

**The engine was choosing what the agent needed to know.** That is the
simulator author's judgement baked into the agent's perception. A judge asking
"where does the agent get that traffic estimate?" gets the answer "we gave it
to him", which is not an answer.

**The simulator is one possible Monterrey, not the truth.** It is a credible
reconstruction, calibrated against real data — but still a reconstruction. An
agent shaped around what that reconstruction happens to offer is fitted to a
fiction.

**The agent could not be exported.** This is the decisive one. Our simulator
is Monterrey-specific by construction: DENUE establishments, INEGI AGEB
population, a TomTom hourly profile for this metro. Drop the agent in
Guadalajara and there is no engine to push it beliefs. It does not degrade —
it does not start.

In production the simulator is replaced by reality, and reality pushes
nothing. It answers questions, if you know which to ask.

---

## 3. The line: what belongs to whom

This is the load-bearing rule of the whole design.

| | May use | Examples |
|---|---|---|
| **Simulator** — stands in for reality | Anything, including Mexico-only sources | DENUE, INEGI AGEB, TomTom Monterrey, Open-Meteo MTY archive |
| **Agent** — exports to any city | Only what a lat/lon and an app screen yield | app feed, weather by coordinate, OSM graph, OSM POIs, its own history |

**The portability test, and it should be enforced by a test, not a
convention:** every input the agent consumes must be obtainable from a pair of
coordinates and a phone screen. Nothing else.

DENUE is Mexico. OpenStreetMap is the planet. Open-Meteo answers any lat/lon.
That is where the cut falls, and it is not a matter of taste.

A corollary worth stating because it is easy to get wrong: the simulator using
DENUE is not cheating. The simulator IS the world. The agent using DENUE would
be cheating, because in Guadalajara it would not have it.

---

## 4. Architecture

```
┌─ AGENT — city-agnostic, exportable ────────────────────────────┐
│                                                                 │
│  RAW SOURCE PORTS        (lat/lon + app screen, nothing more)   │
│    app feed .......... A, B, payout, ETA, km, surge flag,       │
│                        demand heatmap (lagged, coarse, banded)  │
│    weather ........... by coordinate                            │
│    street graph ...... OSM by bounding box                      │
│    POIs .............. OSM restaurants and commerce             │
│    own history ....... empty on arrival, fills every shift      │
│                              |                                  │
│  FEATURE PIPELINE            |   fitted functions, precomputed  │
│    zone quality              |   travel-time model              │
│    hour-of-day rhythm        |   kitchen speed per venue        │
│    demand field              |   app-ETA bias per zone          │
│    risk map by hour          |   return asymmetry               │
│                              |                                  │
│  BELIEF STATE                |   this city, this shift          │
│    every value an Estimate: value + confidence + age            │
│                              |                                  │
│  DECISION                    |   < 7 s. Table reads only.       │
│    reservation price, offer scoring, DecisionTrace              │
│                                                                 │
│  ↑ OFFLINE RE-FIT between shifts, from its own history only     │
└─────────────────────────────────────────────────────────────────┘
```

The agent still imports nothing but `src/core/ports.py`. What changes is the
direction of the call: it **pulls** from raw-source ports instead of being
**pushed** a finished observation.

---

## 5. Online versus offline — the split that makes 7 seconds possible

DiDi gives a courier about seven seconds to accept or reject. That constraint
does not mean the reasoning must be shallow; it means the **fitting** and the
**using** are different activities with different budgets.

| | Online — inside 7 s | Offline — between shifts |
|---|---|---|
| Work | score the offer against fitted functions | re-fit those functions |
| Operations | table reads, arithmetic | regressions, aggregation, clustering |
| Budget | microseconds (measured: 0.33 ms) | minutes, unconstrained |
| Frequency | every offer | once per shift |

This is how a real courier works. Nobody computes a regression at a traffic
light. They decide in seconds using intuitions built over weeks — and those
intuitions are exactly what the offline re-fit produces.

It is also where self-calibration lives. Fitting is expensive; lookup is free.

---

## 6. Cold start: the first shift in an unknown city

The hard question. The agent lands in Puebla with no history.

It starts from **structural priors, never local ones**:

| Prior | Why it transfers |
|---|---|
| Free-flow speed by road class | Physics, read off the OSM `highway` tag |
| Restaurant and commercial density | OSM POIs cover the planet |
| Bimodal meal peaks | Human behaviour; the exact hours are learned |
| Cost of the empty return leg | Geometry |
| Unpaid legs at both ends of a shift | Arithmetic |

Everything else gets a **wide prior with low confidence**. The confidence
discount is already implemented, which means the agent *knows that it does not
know*: it behaves conservatively on shift one and loosens as evidence
accumulates.

**The learning curve is itself a result worth showing.** Run shifts one, two
and three with memory persisting and earnings should climb. "The agent has
never seen this city, and it calibrates itself in three shifts" is a stronger
claim than any single absolute number, because it is the claim that generalises.

A warning against self-deception: shift one should be visibly mediocre. If it
is not, something local leaked into the priors and the portability test is
being violated somewhere.

---

## 7. Variables

### 7.1 What the app gives — the whole set

Six per offer plus two ambient. Adding a field here is a product claim that
the real app shows it.

| Field | Note |
|---|---|
| Point A | pickup coordinate |
| Point B | dropoff coordinate |
| Payout | what the platform offers |
| ETA | the app's estimate — optimistic, and systematically so |
| Kilometres | the app's estimate |
| Surge flag | binary. Never the real multiplier. |
| Demand heatmap | ambient. Lagged, coarse, banded into a few levels. |
| Acceptance rate | ambient. The platform shows it back as pressure. |

### 7.2 What the agent derives for itself

Grouped by the question each answers.

**Is this trip actually profitable?**
- real travel time, from its own model, not the app's ETA
- kitchen wait at this specific branch, from memory
- fuel consumed as **minutes standing still at a pump**, not pesos
- realised payout including locked surge

**Where does it leave me?**
- destination zone quality, from OSM POI density in the dropoff cell
- expected dead minutes after delivering, from believed local demand
- **return asymmetry** — going to San Pedro and coming back are not the same
  cost. One-way streets and limited-access roads are not symmetric, and the
  OSM graph knows it.

**Should I be here at all?**
- hour-of-day rhythm for this city, learned
- competing courier density, inferred from how offers arrive
- surge trajectory: not the level the app shows, but whether it is rising or
  collapsing. The heatmap is lagged, so its derivative is the exploitable part.

**Is it safe?**
- risk by zone and hour, learned
- weather exposure: rain and Monterrey's 40 °C are physical costs, not moods

**Is the shift ending?**
- unpaid distance home, and the growing penalty on orders pointing away
- fuel remaining as minutes
- acceptance rate versus where the platform starts retaliating

### 7.3 The three highest-value additions

Of everything not yet modelled, these carry the most signal per hour of work:

**Destination quality.** A well-paid order that strands you in a residential
desert costs an unpaid return leg. Cheap to compute from OSM POI density and
it changes rankings immediately.

**Return asymmetry.** Real road networks are directional. Treating the trip
back as equal to the trip out systematically undervalues orders that happen to
return along a fast corridor.

**App-ETA reliability, per zone.** The app says eleven minutes; it took
nineteen. After a hundred trips the agent knows the bias per zone better than
the platform will admit. This is the purest arbitrage available and it is
entirely self-learned — it needs no external source at all.

---

## 8. Relationship functions — fit offline, read online

Rather than have the agent derive interactions inside its seven seconds, the
relationships are fitted between shifts and read as tables.

| Function | Fitted from | Read to answer |
|---|---|---|
| `surge_mult(rain, hour)` | own history | is it worth going out in this weather |
| `travel_time(zone, hour)` | own trips, not the app | how long this really takes |
| `kitchen_minutes(venue, hour)` | pickups | peaks are worse than averages |
| `dead_minutes(dest_cell, hour)` | idle after delivering | the true cost of where it leaves me |
| `eta_bias(zone, hour)` | promised versus realised | how much the app is lying here |
| `arrival_rate(cell, hour)` | offers seen | what rejecting this one costs me |

Two properties to preserve:

- **Each table is keyed only by things the agent can observe.** No table may
  be keyed on anything from `src/world/`.
- **Each carries its own sample count and confidence.** A cell visited twice
  must not speak with the authority of one visited two hundred times. This is
  what makes cold start safe rather than reckless.

---

## 9. Consequences, stated honestly

**The numbers probably will not move.** If the agent queries everything it
used to be handed, the information set is identical and so are the decisions.
This change buys defensibility and portability, not performance.

**Performance would only change under a real query budget.** At 0.33 ms
against 7,000 ms, no budget bites naturally. Imposing one artificially would
make the agent *worse*, because it would have less information than today. If
we ever want selecting-which-questions-to-ask to be a genuine decision, the
budget has to come from somewhere real — an API rate limit, a paid call — not
from a number we invented.

**Cold start will look bad, and that is correct.** A first shift in a new city
should underperform. Any design where it does not is leaking local knowledge.

---

## 10. What is measured today

For provenance of every figure below, see the git history — each number is
recorded in the commit that produced it.

**Read §10.1 before quoting any earnings figure.** An earlier version of
this section reported "+23.1% over accept-everything, wins 6/6" as the
headline. That number was real but it was measured on six seeds, and six
seeds cannot carry it.

| | Value |
|---|---|
| Decision latency, SmartPolicy | 1.0 ms median, 18.3 ms p99 |
| Shift generation | 3.3 s for 8 simulated hours |
| Tests | 120 passing |
| Agent import boundary | enforced by AST scan and a clean-subprocess check |
| Agent portability | enforced by 16 tests; verified to fail on a deliberate violation |

Latency was 0.33 ms median / 0.46 ms p99 before the pull inversion. The rise
is the repositioning branch scanning every placeable cell, which grew from
~8 heatmap cells to all 127 when the agent gained `cell_coords`. Memoising
the cell lookup cut p99 from 33.9 ms to 18.3 ms with a byte-identical shift
digest. Against DiDi's ~7,000 ms that is 380× headroom, so this is recorded
as a fact, not a problem.

---

### 10.1 Earnings, and why the seed count decides the story

Reference window 14:00–22:00. **The 6-seed set (42, 7, 13, 5, 21, 99) is
favourable to the agent against accept-everything, and the 18-seed set is
the honest sample.**

| | 6 seeds | 18 seeds |
|---|---|---|
| MXN/h, accept-everything | 91.3 | 94.7 |
| MXN/h, 55-peso payout floor | 115.6 | 104.1 |
| MXN/h, smart | 103.6 | 100.5 |
| smart vs accept-everything | +13.6%, wins 6/6 | **+6.2%, wins 12/18** |
| smart vs the payout floor | −10.4%, wins 1/6 | **−3.5%, wins 7/18** |
| km driven, smart vs the floor | 91.1 vs 175.5 | **85.4 vs 153.9** |

The per-seed margin against the floor spans **−40.2% to +58.3%**. With that
much variance, six draws cannot resolve a mean difference of a few per
cent. Quoting a 6-seed margin as the result is not a rounding choice, it is
a claim the data does not support.

### 10.2 The figure that does not move: pesos per kilometre DRIVEN

Same runs, same window. Total payout over total distance actually driven —
which charges the courier for the distance the platform does not pay for.

| | 6 seeds | 18 seeds |
|---|---|---|
| accept-everything | 4.11 | 4.52 |
| 55-peso payout floor | 5.27 | 5.41 |
| **smart** | **9.10** | **9.42** |
| smart vs the floor | **+73%** | **+74%, wins 18/18** |

**This is the result to state.** Not because it is the largest number
available, but because it is the only one that is stable: +73% on six seeds
and +74% on eighteen, winning every seed in the sample, with a worst case
of +11.3%. The MXN/hour margin moved by 17 points between the same two
samples.

The reason the two metrics disagree is the product's whole thesis. The app
pays by the trip and is silent about the kilometre. Measured on seed 42,
the 13 delivery records sum to 41.3 paid km while the courier drove 82.2 —
**half the driving is unpaid**, and for the payout floor it is 58%. An
agent optimising what the courier actually keeps optimises the kilometre.
An agent optimising gross takings does not.

The honest sentence is therefore **"nearly the same money on 44% less
driving, which is 74% more per kilometre driven"** — and on any single seed
it may well be *less* money. On seed 42, the demo seed, the floor earns
1,158 MXN against smart's 789.

### 10.3 Cold start, with a control arm

12 seeds, 3 consecutive shifts, a different day each shift, history
persisting and `refit()` between them — against a control that runs the
identical days in the identical order and throws the history away.

| | shift 1 | shift 2 | shift 3 |
|---|---|---|---|
| history persisting | 99.2 | 105.8 | 102.6 |
| control, cold every day | 99.2 | 97.7 | 98.2 |
| **delta** | **+0.0** | **+8.1** | **+4.5** |

The control arm is not decoration. Without it the per-seed lines read
90 → 88 → 155 and 104 → 80 → 48, because shift index was confounded with
how hard that particular day was. Uncontrolled, shift 1 → shift 3 looks
like +3.4%; that number means nothing on its own.

By shift 3 the agent holds ~40 trips, 22 travel-correction buckets, 17
ETA-bias zones and 38 remembered kitchens. 7 of 12 seeds finish ahead of
their cold twin.

**The curve is real but shallow, and §6 says to treat that as a leak hunt
rather than a win. The hunt found one.** The agent's own trips measure a
travel correction of 1.12–1.15 — the prior it arrived with was only 12–15%
optimistic, so there was little to learn. That is by construction:
`free_flow_to_scooter_factor` was calibrated **from measured Monterrey
legs**, deliberately, so the cold-start travel belief reproduced the
previous model and the before/after comparison measured one change instead
of two. That traded curve steepness for comparability, and it is a local
prior the agent should not have in an unknown city.

The remaining local priors, listed so nobody has to find them: that scooter
factor; `prior_offers_per_hour = 7.5`; `base_reservation_mxn_per_hour = 20`
and the MXN/km fuel and wear costs; `dead_minutes_at_zero/full_demand`
14/2; the sunrise control points, which are genuinely latitude-dependent.
None is a dataset and all are priors §6 permits — but together they are why
shift one is not visibly bad, which §6 warns is the signature of a leak.

The app-ETA bias is the clean signal by contrast: the agent learns the app
under-promises by 14–29% per zone from nothing but its own completed trips,
with no external source at all.

---

## 11. Known open items, none hidden

- **`self_check` fails on deliveries per hour and MXN per hour, on all
  three policies and all 18 seeds.** Thresholds untouched. The dominant
  cycle component is the ride TO the restaurant.
- **Four of the six relationship functions in §8 are not fittable through
  the current port.** `RawSourcePort` has no `record_offer_seen(cell,
  minute)` and no `record_idle(cell, minutes)`, so `arrival_rate(cell,
  hour)` and `dead_minutes(dest_cell, hour)` cannot be learned per cell;
  `recall_kitchen` takes no minute, so `kitchen_minutes(venue, hour)`
  cannot be keyed by hour; and `record_trip` carries no `promised_km`, so
  the app's distance bias is unlearnable even though its time bias is not.
- **The night-window deficit is a direct consequence of the first of
  those.** The reservation price uses a city-wide lifetime-average arrival
  rate, so it stays at dinner-peak height through three dead hours after
  midnight. The obvious fix — a recency window — measured worse, because
  the same rate also inflates every destination's dead-minute estimate. It
  needs the two uses decoupled, and decoupling them needs a per-cell offer
  history the port cannot currently record.
- **The 12:00–20:00 surge realism window sits at 25.8%, above the 18%
  band.** A flat-profile control run shows 17.8%, so this predates the
  supply profile and lives in the surge constants.
- **The agent supplies its own hour-of-day rhythm prior.** `poi_density_near`
  takes no minute — correct, since POI density is geography — which leaves
  `PlatformView.heatmap` as the only time-varying demand signal reaching
  the agent. So a coarse 12-point bimodal meal prior lives inside the
  agent. It is the one place a prior was added rather than queried for.
