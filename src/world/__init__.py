"""World layer: ground truth for the delivery-courier simulator.

Every module under `src/world/` describes the real, exogenous state of the
simulated city (geography, street network, weather/traffic/order timelines)
or the endogenous state containers a simulation engine mutates tick by tick.

Hard layer boundary: `src/agent/` (a later slice) must NEVER import from
`src/world/` directly. Agents only ever observe the city through a
restricted, noisy view exposed by `src/platform/` and `src/enrichment/`
(also later slices). This keeps the "6 fields the app shows you" vs.
"what the courier actually has access to" distinction real in code, not
just in the pitch.
"""
