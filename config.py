"""
Trail Running Simulator V2
Project-wide configuration.

There are only TWO spatial concepts in the V0 architecture:

1. LEARNING_STEP_M
   The distance between successive starting positions when building the
   historical FIT learning library.

2. GPX_SEGMENT_LENGTH_M
   The normalized GPX segment length.

The GPX segment length is also the historical transition length and the
simulation step length.

Therefore:

    FIT learning:
        1 m rolling starting positions
        -> next GPX_SEGMENT_LENGTH_M

    GPX:
        GPX_SEGMENT_LENGTH_M
        -> GPX_SEGMENT_LENGTH_M
        -> ...

    Simulation:
        one GPX_SEGMENT_LENGTH_M transition at a time
"""


# =============================================================================
# FIT learning density
# =============================================================================

# Historical FIT trajectories are sampled using a rolling starting position
# every 1 metre.
# This is ONLY the density of observations in the historical library.
# It is NOT the simulation step.
# Example with a 100 m GPX segment:
#     0 m   -> next 100 m
#     1 m   -> next 100 m
#     2 m   -> next 100 m
#     3 m   -> next 100 m
#     ...

LEARNING_STEP_M = 1.0


# =============================================================================
# GPX / transition / simulation segment length
# =============================================================================

# This is the normalized GPX segment length.
# The same value defines:
#     - GPX normalization
#     - historical transition length
#     - micro-model target transition
#     - simulation step
# Current V0 calibration:
#     100 m
# To test another granularity, change ONLY this value.
# For example:
#     50.0
#     100.0
#     200.0

GPX_SEGMENT_LENGTH_M = 100.0
