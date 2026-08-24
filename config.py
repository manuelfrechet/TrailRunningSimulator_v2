"""
Project-wide configuration.

There should be one authoritative definition for each project-level
quantity. Modules must import these values rather than hard-code them.
"""

# Historical learning:
# Starting positions are sampled every 1 m.
LEARNING_STEP_M = 1.0

# Historical transition / GPX prediction segment length.
#
# Normal V0:
#     50 m
#
# For the temporary GPX granularity experiment:
#     change this single value to 1000.0
#
# Then restore it to 50.0 afterwards.
TRANSITION_LENGTH_M = 50.0
