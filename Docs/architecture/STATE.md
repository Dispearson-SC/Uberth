# State of play

**Written to survive a context compaction.** Everything a fresh session needs
to pick this up without re-deriving it, including what is broken.

Commit at time of writing: `88de316`. 22 commits, working tree clean, 120
tests passing.

---

## 1. Read these first

| | |
|---|---|
| `Docs/architecture/AGENT_MODEL.md` | what the agent is and why it pulls rather than receives |
| `Docs/architecture/DECISIONS.md` | append-only log D1-D15, including the decisions that were wrong |
| this file | where things actually stand today |

---

## 2. THE STALE ARTEFACT — fix this before demoing anything

**Every file in `replays/` was built before the fare, distance-ceiling and
travel recalibration landed.** 21 recordings plus `days.json`. They play, they
look fine, and the numbers in them describe a world that no longer exists.

Rebuild, in this order, before showing anything:

```
.venv\Scripts\python.exe scripts/build_replays.py     # reference + fork, ~4 min
.venv\Scripts\python.exe scripts/build_duel.py        # hostile day pair,  ~10 min
.venv\Scripts\python.exe scripts/build_days.py        # four days x2,      ~12 min
```

`build_days.py` overwrites `replays/days.json` with the real manifest. The one
on disk right now is PROVISIONAL — hand-written mid-build so the dashboard had
something to show — and it lacks the per-day weather conditions and summaries.

---

## 3. What the simulator now matches, and what it does not

The world was recalibrated against a working courier's own figures. Four of
five now agree.

| | courier reports | simulator | |
|---|---|---|---|
| list price per trip | 30-40 normal, 40-50, 50-80 | 80.0 / 15.9 / 4.1 | ✅ |
| deliveries per hour | 3-4 | 2.5-2.8 | close |
| riding speed over a shift | 30 km/h | 31.8 km/h | ✅ |
| deliveries per shift | 24-32 | 22-24 | ✅ |
| **price AFTER surge** | same 80/15/5 | **66 / 18 / 12 / 4** | ❌ |

The last row is the open one and section 6 says why.

---

## 4. The calibration chain, so nobody re-derives it

Each of these was wrong, and each hid the next. They are listed in the order
they had to be found.

**The fare was fitted to the wrong target.** It was tuned so a whole SHIFT
landed in a plausible earnings range. A shift total can be hit by two errors
that cancel, and it was: 88 MXN a trip at 1.2 deliveries an hour. Only a
per-trip figure exposes that, which is why it took someone who drives.

**The reach radius lied to a busy courier.** The platform offered an order
when the courier was within `reach_km` of its restaurant RIGHT NOW — but a
courier mid-delivery cannot act on "right now". Orders taken on the spot
averaged a 0.70 km ride to the restaurant; orders queued behind work averaged
5.61 km, p90 9.33, max 11.30. `CourierSnapshot` now carries
`finishes_lat/lon/free_at_min` and reach is measured from where the courier
will be FREE. Queueing is untouched and deliberate.

**The world applied car congestion to a motorcycle.** Every multiplier in
`traffic.py` comes from TomTom probes and TomTom probes are cars. D11 already
said a delivery motorcycle filters between lanes — written as the reason the
AGENT does not buy a feed, while the WORLD went on applying car congestion to
a two-wheeler. `MOTORCYCLE_FILTERING_FACTOR = 0.15` scales the EXCESS over
free flow, not the multiplier: filtering buys nothing on an empty road.

**And the one the other three were hiding.** With the cycle finally short, the
courier spent 310 of 500 shift minutes parked with an empty screen — eight
minutes of the entire shift had an offer visible. `baseline_reach_km` went
0.70 -> 1.20, the knee of a sweep, chosen as the smallest radius that is not
starving rather than the largest that scores well.

**A straight-line fare cannot have a flat region.** A courier saying "normal
trips are 30 to 40" is not describing a slope, they are describing the same
price repeated — that is what a minimum fare IS, and no smooth function of km
can imitate it. Three attempts failed before this was obvious.

**Unit error worth remembering.** The card's kilometres are ROAD kilometres —
what the app prints — and `ref_km` is a straight line. Charging the card
against `ref_km` put 91.4% of trips inside the flat minimum, because a trip
the app calls 5 km is 3.4 km as the crow flies. `route_factor` 1.449, measured.

**The geography was solved backwards from the payouts**, on the courier's own
instruction, because their km figures and payout figures could not both hold.
`gravity_d0_km` 1.4 -> 2.6 and `MAX_TRIP_KM` 7.2. The old value carried the
comment "most food delivery in Monterrey is 1-5 km" — an assumption. Stated
consequence: the longest delivery is now 7.2 km, not the 10-12 recalled.

---

## 5. Where the agent stands, and it is not good

**The smart agent now loses to both baselines**, clearly and on every seed
measured.

Its reservation price, thresholds and repositioning were fitted to a world of
few, long, expensive trips. That world was the artefact. In one of short,
frequent, cheap trips, being selective costs more than it earns.

Part of its measured advantage was an advantage over a mis-calibrated world,
not over a baseline. **A result that survives its world being corrected is a
result, and this one did not.** Nothing about the agent should be quoted until
it is re-fitted against the corrected world.

Its constants all point at the old world: `base_reservation_mxn_per_hour`,
`fuel_cost_per_km_mxn`, the dead-minute priors, the offers-per-hour prior.

Earlier figures — "+43% per kilometre driven, 16 of 18 seeds" — were measured
BEFORE the fare and distance work and are now void. Do not quote them.

---

## 6. Open items, none hidden

**Surge shape is wrong, and the ceiling is not why.** 5.3% of orders land in
2.00-2.50 against 2.6% in 1.70-2.00 — the last bucket bigger than the one
before it, where a courier reports surge getting rarer the higher it goes.
Replacing the hard clip with a tanh saturation removed the mass point and
dropped the share above 1.2 surge from 17% to 3.8%, under the 6-18% realism
band; restoring it by raising the gain brought the pile-up straight back. Net
effect nothing, so it was reverted rather than shipped, and the investigation
is recorded in `surge.py` so nobody repeats it. The cause is the DYNAMICS: a
delayed feedback loop past its stability threshold, a bang-bang limit cycle
that spends little time between the rails. Damping it means re-calibrating
migration rate and reaction lag against their own band.

**Post-surge payouts therefore miss the courier's distribution** — 66/18/12/4
against 80/15/5/0. 75% of trips that clear 50 MXN get there on surge, not
distance. The courier's figures describe what the app SHOWS, post-surge; the
card is pre-surge. Those were conflated for most of a session.

**The day window sits at 26.0% above 1.2 surge**, outside the 6-18% band.
Unchanged by any of this work, and not quietly absorbed.

**`self_check` failures changed character but did not vanish.** Every row used
to fail `deliveries_per_hour` at 1.2-1.6 against a [2.0, 3.0] band; that bound
is now mostly met. What fails instead is earning too MUCH (175, 207 against a
150 ceiling) and never being idle (0.02 against a 0.15 floor). The bounds are
the honesty gate and were not loosened.

**The payout floor of 40 sits above the median fare** and refuses more than
half of what it is shown. That is deliberate, chosen by the user: a floor at
or below the median selects nothing, and the point of the rule is to lift the
rate with one number a courier can hold in their head. Recorded in
`agent/calibration.py` so nobody "corrects" it.

**`build_duel.py` obstacle placement is agent-dependent.** Closures are placed
on corridors measured from probe runs of both couriers, so any change to
either policy moves the obstacles and the hostile day stops being comparable
across versions. Fix: freeze the placement from the seed instead.

**Four of six relationship functions in AGENT_MODEL §8 are unfittable** — no
`record_offer_seen`, no `record_idle`, `recall_kitchen` takes no minute,
`record_trip` carries no `promised_km`. The night-window deficit is a direct
consequence of the first.

---

## 7. What the dashboard does

`src/viz/dashboard.html`, opened over a local server (it fetches
`../../replays/`):

```
.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1
http://127.0.0.1:8765/src/viz/dashboard.html
```

- **Day selector** — reference recording, or any day in `days.json`. A day's
  four files load on demand.
- **Hostile day** — swaps BOTH lanes to the same day with five road closures.
  Both lanes always move together: the scenario is a property of the world,
  and showing one courier's storm against the other's clear afternoon is the
  comparison this project already had to correct once.
- **Fork** — a closure scripted into the reference recording only. Disabled,
  not silently inert, on any other scenario.
- **Metrics (or M)** — a modal over the map: money, distance split into paid
  and unpaid, vehicle wear, where the shift's minutes went, offers and
  refusals bucketed by reason, plus a full trip table per courier. Everything
  is computed as of the current scrub minute.

Two honesty rules live in the panel rather than in a reader's head: the wear
figure is labelled as the AGENT's cost model (the world charges fuel in TIME,
never pesos), and below a 40% acceptance rate it states that the platform then
shows about 0.53x the offers — a cost that appears in no figure above it.

---

## 8. The method lesson this session kept re-teaching

Four separate measurement errors, all the same shape: **a silent assumption
about a number, in the place nobody looks.**

- six seeds quoted as a result when the per-seed spread was -40% to +58%
- a matrix diagonal of zero, making same-cell deliveries free
- an assumed 8.0-hour denominator instead of the measured elapsed time
- rain in mm-per-minute compared against thresholds written in mm-per-hour

Every one inflated the agent's numbers. That is not coincidence: **a shortcut
in a calculation tends to fall the way you want it to.**

And the rule that caught most of them: a field name must carry its unit. The
rain defect was not ignorance of the factor — it was that `precip_mm` did not
say which unit, so two layers assumed differently and both stayed
self-consistent. It is `precip_mm_per_hour` now, end to end.
