"""
Trail Running Simulator V2
Project-wide configuration.

This is the single authoritative location for project-level spatial
definitions.

Other modules must import these values rather than hard-code them.
"""


# =============================================================================
# Historical learning resolution
# =============================================================================

# Historical FIT trajectories are converted into rolling transition rows
# starting every 1 metre.
# Example:
#   state at 0 m   -> next transition
#   state at 1 m   -> next transition
#   state at 2 m   -> next transition
# This is independent from the prediction segment length.

LEARNING_STEP_M = 1.0


# =============================================================================
# Transition / prediction segment length
# =============================================================================

# The historical transition attached to each learning position covers this
# length, and the GPX prediction advances by this same length.
#
# Normal V0:
#   50 m
# Temporary experiment:
#   change ONLY this value to 1000.0
# After the experiment, restore it to 50.0.

TRANSITION_LENGTH_M = 100.0
