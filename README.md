# El Repartidor — a courier simulator, and an agent that works a shift in it

Built for the HackMTY 2026 Infosys challenge. Two things live here, and the
separation between them is the point:

- **a world** — Monterrey as a simulated city, with real measured data behind
  every number it produces;
- **an agent** — a courier who works a shift in that world, decides which
  orders to take, and can be moved to another city without changing a line.

The agent never imports the world. It holds a port (`src/core/ports.py`) and
pulls what it needs, the way a real courier pulls at their phone. A test
(`tests/agent/test_portability.py`) fails the build if anything under
`src/agent/` learns a Monterrey-specific fact — a coordinate, a street name,
a venue id. That rule caught the word "Monterrey" in a docstring once, which
is the sort of thing it exists for.

## What is real about the world

| | |
|---|---|
| roads | OpenStreetMap drive network, 95,429 nodes |
| venues | DENUE / INEGI restaurant and workplace census |
| traffic | TomTom Traffic Stats, Monterrey weekday profile, measured |
| weather | Open-Meteo archive, real dates in June-August 2026 |
| geography | H3 resolution-7 cells over the operating polygon |

The fare card, the trip lengths and the delivery rate were calibrated against
a working courier's own figures rather than fitted to a plausible-looking
shift total. `Docs/architecture/STATE.md` records what that changed and what
it broke, including four measurement errors that all happened to inflate the
agent's numbers.

## Running it

Python 3.14, then:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python.exe scripts/build_fixtures.py    # downloads and builds the world
.venv\Scripts\python.exe scripts/run_shift.py         # one shift, headless
.venv\Scripts\python.exe -m pytest -q                 # 132 tests
```

The fixtures are **not** in this repository — the OSM graph and the census
extracts are large and are rebuilt from their sources. `build_fixtures.py`
fetches them. The TomTom step needs a `TOMTOM_API_KEY` in your environment or
a local `.env`; everything else is open data.

## Watching a shift

Recorded shifts play back in a browser with zero live computation, so nothing
can stall on stage:

```
.venv\Scripts\python.exe scripts/build_showcase.py --agent-config Docs/tuning/tuned_agent.json
.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1
```

then open `http://127.0.0.1:8765/src/viz/dashboard.html`. The map shows both
couriers, the app's surge field, the roads closed at that minute and the jam
around them, and — for every minute — the agent's own reasoning in its own
words. Recordings for the showcase days are committed under `replays/`.

## Where the agent actually stands

Honestly, because a result that needs a friendly reading is not a result.

The agent was re-fitted after the world was recalibrated, measured over four
real weekdays and twelve seeds, with the search seeds and the hold-out seeds
kept apart. Against a flat 40 MXN payout floor, on net earnings per hour:

| | vs the floor | wins |
|---|---|---|
| before re-fitting | 79.0% | 4 / 26 |
| after | 97.8% | 10 / 26 |

On clean days, like for like, it wins two of three and earns more per
kilometre in every scenario measured. Over all twenty-six it is still about
two per cent behind.

The reason is worth more than the number. Offer quality in this world spreads
2.88x, but the thing that decides it — kitchen wait, which varies four-fold
and correlates 0.003 with distance — is invisible on the offer card. A
threshold on payout is close to the best available strategy when the only
reliable signal is the payout. The agent is not losing because it cannot
think; it is losing because the simulator hides what it would think about.
That is an open item, and it is a world problem, not an agent problem.

## Deploying the dashboard

There is no server-side application to deploy. A shift is simulated offline,
every tick is recorded to a file, and the browser plays it back with zero
computation on the server — which is the whole reason the demo cannot stall
on stage. So the deploy is nginx plus 14 MB of recordings.

A `Dockerfile` is included. On Coolify: **New Resource → Docker Compose /
Dockerfile**, point it at this repository, build pack `Dockerfile`, port
`80`. Nothing else is required — no environment variables, no volumes, no
database. `/` redirects to the dashboard.

The nginx config in `deploy/nginx.conf` sends `no-store` on the HTML and
caches the JSON for an hour. That split is deliberate: the recordings are
immutable once written, and a cached copy of the PAGE is the thing that
wastes an afternoon, because whoever is looking at the stale version is
rarely the person who changed it.

## Documentation

- `Docs/architecture/STATE.md` — where things stand, including what is broken
- `Docs/architecture/AGENT_MODEL.md` — what the agent is, and why it pulls
- `Docs/architecture/DECISIONS.md` — an append-only log, including the
  decisions that turned out to be wrong

## License

Copyright (C) 2026 Gerardo Tapia.

This program is free software: you can redistribute it and/or modify it under
the terms of the **GNU Affero General Public License** as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
details.

You should have received a copy of the GNU Affero General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.

The AGPL's section 13 is the clause that matters for something like this: if
you run a modified version of this software so that other people interact with
it over a network, those users must be able to get its source.

The data this project consumes is not covered by that license and carries its
own terms — OpenStreetMap (ODbL), DENUE/INEGI, TomTom, and Open-Meteo each set
their own.
