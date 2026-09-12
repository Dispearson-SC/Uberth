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

| | Value |
|---|---|
| Decision latency, SmartPolicy | 0.33 ms median, 0.46 ms p99 |
| Shift generation | 3.3 s for 8 simulated hours |
| Tests | 88 passing |
| Agent import boundary | enforced by AST scan and a clean-subprocess check |
| Smart versus accept-everything, reference window | +23.1% mean, wins 6/6 seeds |
| Smart versus a 55-peso payout floor, reference window | −2.8% mean, wins 3/6 seeds |
| Kilometres driven, smart versus that floor | ~102 km versus ~176 km |

The honest headline is **not** "earns more". It is **"earns nearly the same on
42% less driving"** — 74 fewer kilometres per shift of fuel, wear, risk and
Monterrey heat that the courier does not pay for.

Known open items, none hidden:

- `self_check` fails on deliveries per hour and MXN per hour. Thresholds
  untouched. The dominant cycle component is the ride TO the restaurant.
- The night window loses 22% to accept-everything. Diagnosed: the reservation
  price uses a lifetime-average arrival rate, so it stays at dinner-peak
  height through three dead hours after midnight. The obvious fix — a
  recency-windowed rate — measured worse, because the same rate also inflates
  every destination's dead-minute estimate. It needs the two uses decoupled.
- The 12:00-20:00 surge realism window sits at 25.8%, above the 18% band. A
  flat-profile control run shows 17.8%, so this predates the supply profile
  and lives in the surge constants.
