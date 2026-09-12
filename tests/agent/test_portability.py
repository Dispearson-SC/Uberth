"""The agent must work in a city we never calibrated anything for.

This is the test the whole push-to-pull inversion exists to make possible,
and it is the one a judge can check in ten seconds. `Docs/architecture/
AGENT_MODEL.md` section 3 states the rule:

    Simulator — stands in for reality: may use anything, including
                Mexico-only sources.
    Agent     — exports to any city: only what a lat/lon and an app screen
                yield.

The corollary that is easy to get backwards: the SIMULATOR using DENUE is
not cheating, because the simulator IS the world and the world is allowed to
be Mexican. The AGENT using DENUE would be cheating, because in Guadalajara
it would not have it.

WHAT IS SCANNED, and why it is this set. `src/agent/` is the decision layer.
The raw-source adapter is the agent's perception surface — the object that
gets replaced by real APIs in production — so its own modules are held to
the same rule: if the QUERIES it exposes are shaped around a country's
dataset, swapping the implementation does not save the agent.

What is NOT scanned, deliberately: the estimate builders behind that surface
(`demand_sense`, `traffic_estimate`, `weather_estimate`,
`events_perception`) and `EnrichmentAdapter`. Those are the simulator
standing in for reality, they keep their honest provenance comments naming
the real datasets they read, and that provenance is a feature. The line
falls at the interface, which is exactly what the signature check below
pins down.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from src.core.ports import RawSourcePort
from src.enrichment.raw_source import RawSourceAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = PROJECT_ROOT / "src" / "agent"

# The agent's own perception surface: the `RawSourcePort` implementation and
# the modules it owns. Anything reachable BEHIND these is the simulator.
RAW_SOURCE_MODULES = (
    PROJECT_ROOT / "src" / "enrichment" / "raw_source.py",
    PROJECT_ROOT / "src" / "enrichment" / "osm_travel.py",
    PROJECT_ROOT / "src" / "enrichment" / "history.py",
)

# Dataset names, column names and place codes that exist only because this
# happens to be Monterrey, Nuevo Leon, Mexico. Any of these in the scanned
# set means the agent (or its perception surface) has grown a dependency it
# cannot take to another city.
FORBIDDEN_IDENTIFIERS = (
    "denue",
    "inegi",
    "ageb",
    "tomtom",
    "scian",
    "cve_ent",
    "cve_mun",
    "cve_loc",
    "nom_estab",
    "nombre_act",
    "codigo_act",
    "per_ocu",
    "manzana",
    "cod_postal",
    "municipio",
    "monterrey",
    "nuevo_leon",
    "nuevoleon",
    "demo_downtown",
    "restaurants.parquet",
    "population.parquet",
    "workplaces.parquet",
    "cells.parquet",
    "travel_matrix.npz",
)

# `restaurant_denue_id` is an OPAQUE VENUE KEY and is explicitly allowed:
# it is whatever stable identifier the platform prints next to the branch
# name, a courier plainly sees which branch they are being sent to, and
# nothing in the agent parses it or resolves it against a registry. Every
# market's app has some such key. Loading `denue_19_csv` is the violation;
# remembering that venue #4471 is always slow is not.
ALLOWED_VENUE_KEY_TOKENS = ("restaurant_denue_id", "denue_id")

# Every parameter a `RawSourcePort` method may take. This is the portability
# rule stated as a signature: a coordinate, a clock, a reach, a cell id the
# port itself handed back, an opaque venue key, and measurements of a trip
# the courier just made. Nothing else — because nothing else is available
# from a pair of coordinates and a phone screen.
ALLOWED_PORT_PARAMETERS = {
    "self",
    "lat",
    "lon",
    "from_lat",
    "from_lon",
    "to_lat",
    "to_lon",
    "minute",
    "radius_km",
    "cell",
    "cells",
    "from_cell",
    "to_cell",
    "denue_id",  # the opaque venue key; see ALLOWED_VENUE_KEY_TOKENS
    "observed_minutes",
    "promised_minutes",
    "actual_minutes",
    "actual_km",
}

# Hardcoded coordinates are the leak a token scan misses: an agent with
# 25.67 and -100.31 baked into it is pinned to one city as surely as one
# loading a Mexican census, and no name-based check would see it.
#
# Two bands, and the latitude one needs a second condition to be useful: a
# calibration constant can legitimately be 25.0 minutes or a 30.0 km reach,
# so a bare magnitude in the latitude range proves nothing. A COORDINATE
# carries precision — nobody writes a position to the whole degree — so the
# latitude test also requires at least two decimal places. The longitude
# band needs no such qualifier: a negative number around -100 has no other
# business in a decision layer.
LATITUDE_BAND = (20.0, 35.0)
LATITUDE_MIN_DECIMALS = 2
LONGITUDE_BAND = (-115.0, -90.0)


def _decimals(value: float) -> int:
    text = repr(float(value))
    if "." not in text or "e" in text:
        return 0
    tail = text.split(".", 1)[1]
    return 0 if tail == "0" else len(tail)


def _looks_like_a_coordinate(value: float) -> bool:
    if LONGITUDE_BAND[0] <= value <= LONGITUDE_BAND[1]:
        return True
    return (
        LATITUDE_BAND[0] <= value <= LATITUDE_BAND[1]
        and _decimals(value) >= LATITUDE_MIN_DECIMALS
    )


def scanned_files() -> list[Path]:
    files = sorted(AGENT_DIR.rglob("*.py")) + [p for p in RAW_SOURCE_MODULES]
    assert files, "expected python modules to scan"
    for path in files:
        assert path.exists(), "scanned file is missing: %s" % path
    return files


def _referenced_names(path: Path) -> set[str]:
    """Every name and attribute the CODE in a module refers to.

    Read off the AST rather than the text on purpose: these modules name the
    forbidden paths in their own docstrings, precisely in order to say that
    they are forbidden. A grep cannot tell a prohibition from a violation;
    a parse can.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def _strip_allowed(text: str) -> str:
    for token in ALLOWED_VENUE_KEY_TOKENS:
        text = text.replace(token, "<venue-key>")
    return text


@pytest.mark.parametrize("path", scanned_files(), ids=lambda p: p.name)
def test_no_city_specific_identifier_reaches_the_agent_or_its_sources(path: Path) -> None:
    """The agent and its perception surface name no country's dataset.

    Scanned over the whole file, comments included, on purpose: a comment
    saying "read from the DENUE establishment fixture" is a true statement
    about a dependency, and a dependency is what this test is looking for.
    """
    text = _strip_allowed(path.read_text(encoding="utf-8").lower())
    for token in FORBIDDEN_IDENTIFIERS:
        assert token not in text, (
            "%s references %r. That belongs to the SIMULATOR, which stands in for "
            "reality and may use Mexico-only sources; the agent exports to any city "
            "and may not. See Docs/architecture/AGENT_MODEL.md section 3."
            % (path.name, token)
        )


def test_the_agent_has_no_coordinates_of_its_own() -> None:
    """No latitude or longitude literal anywhere under `src/agent/`.

    Every coordinate the decision layer touches arrives from the port or
    from the courier's own snapshot. One baked in would pin the agent to
    this metro just as firmly as loading a census would, and would not trip
    any name-based check.
    """
    for path in sorted(AGENT_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, float):
                continue
            assert not _looks_like_a_coordinate(node.value), (
                "%s line %d contains %r, which reads as a coordinate in this "
                "metro's band. The agent must RECEIVE every coordinate it uses, "
                "never hold one." % (path.name, node.lineno, node.value)
            )


def test_every_raw_source_query_is_answerable_from_a_latlon_and_a_screen() -> None:
    """The portability rule, as a signature check on the port itself.

    A method that needed a municipality code, a census tract, a dataset
    path or a world object would show up here as a parameter outside the
    allowed set — which is the point: an interface is portable or it is not,
    and that is decidable without reading any implementation.
    """
    methods = [
        name
        for name in dir(RawSourcePort)
        if not name.startswith("_") and callable(getattr(RawSourcePort, name, None))
    ]
    assert methods, "expected RawSourcePort to declare methods"
    for name in methods:
        signature = inspect.signature(getattr(RawSourcePort, name))
        for parameter in signature.parameters:
            assert parameter in ALLOWED_PORT_PARAMETERS, (
                "RawSourcePort.%s takes %r, which is not obtainable from a pair of "
                "coordinates and an app screen. Every query the agent makes has to "
                "be answerable in a city we know nothing else about."
                % (name, parameter)
            )


def test_the_adapter_actually_implements_that_port() -> None:
    """A portable interface nobody implements proves nothing."""
    missing = [
        name
        for name in dir(RawSourcePort)
        if not name.startswith("_")
        and callable(getattr(RawSourcePort, name, None))
        and not callable(getattr(RawSourceAdapter, name, None))
    ]
    assert not missing, "RawSourceAdapter does not implement %s" % missing


def test_perceived_disruptions_comes_only_from_the_detectability_filter() -> None:
    """The honesty property, pinned where it can actually be checked.

    `perceived_disruptions` must be built exclusively from
    `src.world.events.perceivable_events`. Never `active_events`, never the
    raw timeline, never `Event.is_active` or `Event.start_min`. If a crash
    starts at minute 143 and is detectable from 149, the agent must not see
    it at 145 — and the only reason it cannot is that the adapter has no
    other path to an event.
    """
    # The adapter itself may not touch an event at all beyond holding the
    # list to hand on: no lifetime, no timing, no visibility.
    for path in RAW_SOURCE_MODULES:
        for name in _referenced_names(path):
            assert name not in {
                "active_events",
                "is_active",
                "start_min",
                "end_min",
                "detectable_from_min",
                "perceivable_events",
            }, (
                "%s touches event timing through %r. The adapter's only permitted "
                "path to an event is `build_perceived_events`, which goes through "
                "`perceivable_events`, which is where detectability is applied."
                % (path.name, name)
            )

    # And the one module that IS allowed to touch the timeline decides
    # nothing itself: it calls the filter, and it may read
    # `detectable_from_min` off what comes back to age the estimate — that
    # is a consequence of visibility, not a decision about it.
    translator = PROJECT_ROOT / "src" / "enrichment" / "events_perception.py"
    names = _referenced_names(translator)
    assert "perceivable_events" in names, (
        "events_perception.py must reach events through `perceivable_events`."
    )
    for name in ("active_events", "is_active", "start_min"):
        assert name not in names, (
            "events_perception.py decides visibility itself via %r; that is "
            "`perceivable_events`'s job and nobody else's." % name
        )


# Methods that answer with a reading rather than a fact, a geography lookup
# or an acknowledgement. Every one of these must hand back an `Estimate`, so
# the agent can never be given certainty it does not have.
ESTIMATE_RETURNING = (
    "weather_at",
    "travel_estimate",
    "congestion_near",
    "poi_density_near",
    "recall_kitchen",
    "recall_eta_bias",
)


def test_every_reading_comes_back_as_an_estimate_never_a_bare_float() -> None:
    """A source that returns a bare number is a source claiming certainty.

    These are readings a courier took, not facts handed down, and the
    difference is load-bearing: the confidence discount is the whole reason
    the agent behaves conservatively on a first shift instead of acting
    confidently on a prior it has no evidence for. A leak here would not
    show up in any single number — only as an agent that is mysteriously
    good.
    """
    for name in ESTIMATE_RETURNING:
        annotation = str(inspect.signature(getattr(RawSourcePort, name)).return_annotation)
        assert "Estimate" in annotation, (
            "RawSourcePort.%s returns %s. Every reading is an Estimate with a "
            "confidence and an age, never a bare float." % (name, annotation)
        )


def test_the_hand_built_fake_satisfies_the_same_port() -> None:
    """The policy tests would be worthless against a fake with a different
    shape: they would pass on an interface the real adapter does not have."""
    from tests.agent.fakes import FakeRawSource

    fake = FakeRawSource()
    for name in dir(RawSourcePort):
        if name.startswith("_") or not callable(getattr(RawSourcePort, name, None)):
            continue
        assert callable(getattr(fake, name, None)), "FakeRawSource is missing %s" % name
