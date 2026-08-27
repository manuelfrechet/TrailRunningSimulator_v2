# Trail Running Simulator V0 Specification

## Status of this specification

This document is the authoritative project specification for Trail Running
Simulator V0.

The rules and architecture defined here are FROZEN unless the user explicitly
changes them.

Implementation convenience, previous assistant suggestions, experimental code,
or assumptions made during development do not override this specification.

When existing code conflicts with this specification, the conflict must be
identified explicitly before changing the implementation.


---

# PART I — DEVELOPMENT AND SCIENTIFIC RULES


## 1. Never assume

Do not introduce a modelling, mathematical, statistical, architectural,
preprocessing, interpolation, filtering, smoothing, weighting, clipping,
thresholding, normalization, or simulation decision unless:

1. it is already explicitly defined in this SPEC.md; or
2. it has been explicitly proposed, discussed, and approved by the user.

If a required decision is not specified, stop and ask.

Implementation convenience is never sufficient justification for changing
the model.

Code must implement the specification.

The specification must not be silently changed to accommodate the code.


---

## 2. Scientific method

Development must follow a methodical and scientific approach.

For every important modelling decision:

1. identify the assumption;
2. explain why it is required;
3. identify plausible alternatives when relevant;
4. test the reasoning;
5. use empirical evidence where possible;
6. distinguish observations from interpretations;
7. actively guard against confirmation bias;
8. do not tune a method merely because it produces the expected answer.

Agreement with the user is not the objective.

Accuracy, internal consistency, interpretability, and evidence are the
objectives.

If an assumption, hypothesis, conclusion, or proposed mechanism appears weak
or incorrect, it must be challenged explicitly and the reason explained.


---

## 3. One conceptual change at a time

During diagnosis, calibration, and model development, change one conceptual
element at a time whenever reasonably possible.

Do not simultaneously alter several modelling assumptions and then attribute
the result to one of them.

When fixing a coding error, do not silently change model methodology.

When changing model methodology, identify explicitly what is being changed.


---

## 4. Evidence over implementation

Unexpected model output is diagnostic information.

When evidence contradicts the current implementation:

- stop;
- inspect the cause;
- challenge the current assumption;
- do not defend an implementation merely because it was previously proposed.

A program running successfully does not prove that its modelling assumptions
are correct.


---

## 5. Do not optimize for the expected answer

Known race distance, published elevation gain, historical race time, or other
external information may be used as validation evidence.

It must not automatically become a calibration target.

If a method produces the expected value, investigate why.

If it does not produce the expected value, investigate why.

Do not tune parameters solely until the expected number appears.


---

## 6. Challenge assumptions

When a modelling idea is proposed:

- identify its assumptions;
- test the reasoning;
- identify relevant counterarguments;
- consider alternative interpretations;
- distinguish known facts from hypotheses.

Do not agree merely because a proposal sounds plausible.

Conversely, once an architectural decision has been explicitly made and frozen,
do not repeatedly reopen it without new evidence.


---

## 7. Repository as project memory

Do not rely solely on conversational memory for frozen project decisions.

This SPEC.md is the persistent project contract.

When there is uncertainty about an already-agreed project rule, consult this
specification rather than inventing or re-deriving the rule.

If the conversation and SPEC.md appear to conflict, identify the conflict
explicitly before changing the implementation.


---

## 8. Core code versus diagnostics

The operational model and the diagnostic environment must remain separated.

Core operational modules are:

- config.py
- fit_learning.py
- macro_model.py
- micro_model.py
- gpx_profile.py
- simulator.py
- app.py

Temporary analysis, validation, calibration, experimental charts, diagnostic
tables, model-development checks, and similar research tools belong in:

- learning_diagnostics.py

The design objective is:

    DELETE learning_diagnostics.py
        +
    remove its import/call from app.py
        =
    operational simulator continues to work normally

Diagnostics must never become a dependency of the core prediction pipeline.


---

## 9. Keep app.py clean

app.py is an orchestration and user-interface layer.

It may:

- collect user inputs;
- call operational modules;
- display operational results;
- call learning_diagnostics.py when diagnostics are enabled.

It should not contain:

- FIT learning algorithms;
- macro-model mathematics;
- micro-model mathematics;
- GPX processing algorithms;
- simulation algorithms;
- temporary validation algorithms;
- calibration experiments.

Those belong in their dedicated modules.


---

## 10. Central configuration

Project-wide parameters must have one authoritative definition.

They belong in:

    config.py

Do not duplicate project-level constants across modules.

V0 currently has exactly TWO spatial configuration concepts:

    LEARNING_STEP_M
    GPX_SEGMENT_LENGTH_M

There is no separate hidden prediction length, transition length, or simulator
step length.


---

## 11. Code-change discipline

Before changing an existing module:

1. inspect the current module;
2. identify the exact requested change;
3. check this SPEC.md;
4. identify affected interfaces;
5. change only what is required.

Do not reconstruct an existing module from memory when the current source can
be inspected.

Do not introduce unrelated refactoring during a modelling change unless it is
necessary and explicitly identified.


---

## 12. Full-module delivery

When a change materially affects several parts of a module, provide the complete
replacement module rather than requiring the user to manually combine many
patches.

For trivial and isolated changes, a precise small replacement may be used.

The objective is to minimize integration errors and ambiguity.


---

## 13. Temporary code

Temporary validation or calibration code must be explicitly identifiable and
isolated in learning_diagnostics.py.

Before finalizing V0:

- temporary diagnostic code must be removable;
- obsolete experimental paths must be removed;
- stale configuration variables must be removed;
- comments must describe the final architecture rather than historical
  experiments.


---

# PART II — MODEL SPECIFICATION


## 14. Objective

The simulator predicts elapsed time for every normalized segment of a future
GPX race.

The segment length is defined globally by:

    GPX_SEGMENT_LENGTH_M

Current V0 configuration:

    GPX_SEGMENT_LENGTH_M = 100.0

The primary outputs are:

- predicted time for each normalized segment;
- predicted cumulative time;
- predicted arrival time at each aid station;
- predicted total race time.

The simulator does not need to predict HR, cadence, power, cardiovascular debt,
mechanical debt, or neuromuscular debt as primary outputs.


---

## 15. Spatial architecture

V0 has exactly two spatial concepts.


### 15.1 FIT learning density

Historical FIT races are limited in number.

The expected final historical corpus may contain only approximately 5–10 races.

To artificially increase the historical observation library, FIT trajectories
are sampled using rolling spatial starting positions.

The spacing between successive historical starting positions is:

    LEARNING_STEP_M

Current V0 configuration:

    LEARNING_STEP_M = 1.0

Therefore historical observations begin at:

    0 m
    1 m
    2 m
    3 m
    ...

The 1 m value describes observation density only.

It is NOT the simulator step.


### 15.2 GPX segment length

The future GPX is normalized using:

    GPX_SEGMENT_LENGTH_M

Current V0 configuration:

    GPX_SEGMENT_LENGTH_M = 100.0

This same value defines:

- GPX normalization segment length;
- historical FIT transition horizon;
- micro-model target horizon;
- simulation step.

Therefore, with the current V0 configuration, historical FIT learning creates:

    state at 0 m -> observed next 100 m
    state at 1 m -> observed next 100 m
    state at 2 m -> observed next 100 m
    ...

while the future simulation proceeds:

    0 -> 100 m
    100 -> 200 m
    200 -> 300 m
    ...


---

## 16. Two independent prediction models

The simulator contains two independent models.


### 16.1 Macro model

The macro model learns the relationship between:

- distance from start;
- cumulative ascent;
- cumulative descent;

and cumulative elapsed race time.

Conceptually:

    T_macro =
        M(
            distance,
            cumulative_ascent,
            cumulative_descent
        )

The macro model provides an independent macro-scale prediction.

It does not modify the micro model.


### 16.2 Micro model

The micro model predicts elapsed time for the next:

    GPX_SEGMENT_LENGTH_M

For a prediction position x, the state vector is:

1. distance from start;
2. cumulative ascent from start;
3. cumulative descent from start;
4. cumulative elapsed predicted time;
5. ascent of the coming segment;
6. descent of the coming segment;
7. average grade of the coming segment.

Conceptually:

    delta_t_segment =
        F(state)

The historical FIT corpus provides observations of this relationship.

The micro model is empirical / analogue based.


---

## 17. Historical FIT learning

All uploaded FIT files are processed together.

The historical FIT corpus is the source of learning.

FIT learning is not required to use the same preprocessing methodology as the
future GPX.

Learning is learning.

The GPX is the future trajectory to which the learned behaviour is applied.


### 17.1 Rolling historical observations

For every FIT:

- reconstruct the historical trajectory;
- use rolling starting positions separated by LEARNING_STEP_M;
- for every position d where d + GPX_SEGMENT_LENGTH_M exists, create one
  historical transition.

With the current configuration:

    LEARNING_STEP_M = 1 m
    GPX_SEGMENT_LENGTH_M = 100 m

the historical transitions are:

    0 -> 100 m
    1 -> 101 m
    2 -> 102 m
    3 -> 103 m
    ...


### 17.2 Historical transition state

Each historical transition contains:

- distance from start;
- cumulative ascent;
- cumulative descent;
- elapsed time from start;
- ascent over the next GPX_SEGMENT_LENGTH_M;
- descent over the next GPX_SEGMENT_LENGTH_M;
- average grade over the next GPX_SEGMENT_LENGTH_M;
- actual elapsed time over the next GPX_SEGMENT_LENGTH_M.


### 17.3 Historical target

The target is:

    actual_segment_time_s =
        time_at_d_plus_segment
        -
        time_at_d

where:

    segment = GPX_SEGMENT_LENGTH_M


### 17.4 Combined historical corpus

All uploaded FIT races contribute to ONE combined historical learning corpus.

The micro model learns from the combined trajectory library, not from one FIT
file at a time.


---

## 18. Stationary time in historical learning

Historical stationary periods are not automatically removed from the FIT
learning corpus.

A stop naturally produces unusually slow historical transitions.

These observations remain part of the historical trajectory library.

The current V0 hypothesis is that ordinary future terrain states will generally
not select stop-contaminated historical transitions as their closest analogues.

Do not add an explicit stationary variable, stationary filter, or stationary
weighting unless later evidence shows that it is necessary.

Stationary-time diagnostics may remain available in learning_diagnostics.py.


---

## 19. Macro model

The macro model learns cumulative elapsed time as:

    M(
        distance_from_start,
        cumulative_ascent,
        cumulative_descent
    )


### 19.1 Physical origin

The macro model is structurally anchored at:

    M(0, 0, 0) = 0

There is no free intercept.


### 19.2 Current V0 mathematical form

The V0 macro model uses:

- linear terms;
- quadratic terms;
- pairwise interaction terms.

The macro model remains independent from the micro model.


### 19.3 Macro segment prediction

For a normalized segment:

    X_k -> X_k+1

the unconstrained macro segment prediction is:

    raw_macro_segment_time =
        M(X_k+1)
        -
        M(X_k)


### 19.4 Macro physical constraint

The unconstrained polynomial approximation may occasionally produce:

    M(X_k+1) < M(X_k)

which implies a physically impossible negative segment duration.

For V0, the simulator applies:

    macro_segment_time =
        max(
            raw_macro_segment_time,
            0
        )

The raw unconstrained value must remain available for diagnostics.

The number of clipped macro segments and the magnitude of the correction must
remain measurable.

Clipping must not silently conceal widespread macro-model instability.


---

## 20. Micro analogue search

A future GPX prediction state is compared with the complete historical state
library.

"Closest" means the historical state whose complete multivariable state is
most similar.

It does NOT mean geographically closest distance.


### 20.1 State vector

The seven V0 state dimensions are:

1. distance_from_start_m
2. cumulative_ascent_m
3. cumulative_descent_m
4. elapsed_time_s
5. segment_ascent_m
6. segment_descent_m
7. segment_grade_pct


### 20.2 Standardization

For V0, state distance is standardized Euclidean distance.

Each variable is standardized using statistics from the historical corpus.

Conceptually, for historical state H and query state Q:

    D(H,Q) =
        sqrt(
            sum(
                ((Q_j - H_j) / sigma_j)^2
            )
        )


### 20.3 Number of analogues

The two historical states with the smallest standardized Euclidean distance are
selected.

The number of analogues is fixed at two for V0.


---

## 21. Micro prediction

The two closest historical transitions provide two observed next-segment times.

Let:

- D1 = distance to closest state;
- D2 = distance to second closest state;
- t1 = observed next-segment time for closest state;
- t2 = observed next-segment time for second closest state.

V0 uses inverse-distance weighting:

    t_pred =
        (
            t1 / (D1 + epsilon)
            +
            t2 / (D2 + epsilon)
        )
        /
        (
            1 / (D1 + epsilon)
            +
            1 / (D2 + epsilon)
        )

If the closest distance is effectively zero, use the observed time of the
closest state directly.

Do not add:

- macro correction;
- hand-tuned feature weights;
- clipping;
- extra state variables;
- filtering;

unless explicitly discussed and approved.


---

## 22. GPX normalization

The future GPX is normalized into segments of:

    GPX_SEGMENT_LENGTH_M

Current V0:

    100 m


### 22.1 Raw GPX distance

Raw GPX latitude and longitude are used to calculate cumulative horizontal
course distance.

Latitude and longitude are spatial parsing inputs.

They are not model state variables.


### 22.2 Normalized boundaries

Normalized boundaries are created at:

    0
    GPX_SEGMENT_LENGTH_M
    2 * GPX_SEGMENT_LENGTH_M
    3 * GPX_SEGMENT_LENGTH_M
    ...

With the current configuration:

    0
    100
    200
    300
    ...


### 22.3 Elevation interpolation

ONLY elevation is interpolated from the raw GPX onto normalized boundaries.

At a normalized boundary:

- if an existing raw GPX point corresponds to that distance, use its elevation;
- otherwise find the two surrounding raw GPX points;
- linearly interpolate elevation between those two points.

Do NOT:

- resample the GPX onto a 1 m grid;
- accumulate raw GPX ascent/descent before normalization;
- interpolate raw cumulative ascent/descent;
- use polynomial interpolation;
- introduce smoothing or filtering unless explicitly approved.


### 22.4 Terrain calculation

Terrain quantities are calculated AFTER elevation normalization.

For consecutive normalized elevations:

    delta_elevation =
        elevation_end
        -
        elevation_start

Then:

    ascent_m =
        max(
            delta_elevation,
            0
        )

    descent_m =
        max(
            -delta_elevation,
            0
        )

Cumulative ascent and descent are then calculated from the normalized segment
values.

Segment grade is:

    grade_pct =
        (
            elevation_end
            -
            elevation_start
        )
        /
        GPX_SEGMENT_LENGTH_M
        * 100


### 22.5 Current empirical observation

During development, raw GPX elevation granularity produced substantially
inflated cumulative ascent when every small raw elevation fluctuation was
accumulated.

Using a 100 m normalized elevation representation produced approximately
6151 m cumulative ascent for the tested SwissPeak course, close to the
approximately 6000 m published race value.

This is evidence supporting the current 100 m V0 representation.

The published value remains validation evidence, not an automatic calibration
target.


---

## 23. Simulation

Simulation proceeds exactly one:

    GPX_SEGMENT_LENGTH_M

at a time.

With the current configuration:

    100 m -> 100 m -> 100 m -> ...


### 23.1 Simulation sequence

For each segment:

1. determine the state at the segment start;
2. read the terrain of the coming normalized GPX segment;
3. calculate the independent macro prediction;
4. construct the seven-dimensional micro query;
5. search the historical analogue library;
6. calculate the micro segment prediction;
7. store both predictions;
8. update cumulative macro time;
9. update cumulative micro time;
10. apply explicitly defined aid-station stop time where applicable;
11. move to the next normalized GPX segment.


### 23.2 Micro elapsed-time recursion

The elapsed_time_s supplied to the micro query is the cumulative simulated
micro elapsed time available at the start of that segment.

Therefore:

    micro_elapsed_time_(k+1)
        =
    micro_elapsed_time_k
        +
    micro_predicted_segment_time_k
        +
    applicable_stop_time

The simulator does not know the actual future race elapsed time.

It must use its own previously simulated micro trajectory.


---

## 24. Macro and micro independence

Macro and micro are independent estimates.

Do not:

- force micro toward macro;
- force macro toward micro;
- automatically average them;
- calibrate one using the other;
- use their difference to automatically correct either model.

Differences between macro and micro are information to be investigated.

They are not automatically errors.


---

## 25. Aid stations

Each aid station may contain:

- aid station name;
- distance from race start;
- expected stop duration in minutes.

Aid-station stop duration represents elapsed clock time spent stopped.

The stop duration is added independently to both macro and micro cumulative
clocks.

For both models:

    cumulative_time_after_stop =
        cumulative_time_before_stop
        +
        stop_duration

Aid-station stop time is not treated as running time.

Arrival time at the aid station is recorded before adding the stop.

Departure time is recorded after adding the stop.


---

## 26. Main output

One row represents one normalized GPX segment.

Required terrain columns:

- distance_from_start_m
- ascent_m
- descent_m
- cumulative_ascent_m
- cumulative_descent_m
- grade_pct

Required macro/micro prediction columns:

- macro_predicted_time_s
- micro_predicted_time_s
- segment_difference_s
- macro_predicted_cumulative_time_s
- micro_predicted_cumulative_time_s
- cumulative_difference_s

Macro diagnostic fields may additionally include:

- raw_macro_predicted_time_s
- macro_time_clipped

Aid-station columns:

- aid_station_name
- aid_station_stop_min

Human-readable time columns may additionally be provided.


---

## 27. Macro / micro comparison

The output must show macro and micro predictions side by side.

For each segment:

    segment_difference_s =
        micro_predicted_time_s
        -
        macro_predicted_time_s

For cumulative time:

    cumulative_difference_s =
        micro_predicted_cumulative_time_s
        -
        macro_predicted_cumulative_time_s

These differences are diagnostic only.

They must never be used to automatically modify either model.

The user decides whether the model is valid.


---

## 28. Diagnostics architecture

All model-development diagnostics belong in:

    learning_diagnostics.py

This includes, when required:

- historical learning diagnostics;
- stationary-time diagnostics;
- macro historical checks;
- leave-one-FIT-out validation;
- analogue diagnostics;
- cumulative macro-versus-micro divergence analysis;
- macro clipping analysis;
- calibration experiments;
- temporary charts and tables.

app.py may call diagnostic rendering functions.

The diagnostic module must not be required for the operational simulator to
function.


---

## 29. Model development methodology

The model is developed incrementally.

For each experiment:

- use the same complete FIT corpus when comparison requires it;
- start with the simplest defensible model;
- add or change exactly one variable or mechanism when reasonably possible;
- rebuild the relevant model;
- compare the resulting predictions;
- inspect unexpected behaviour rather than automatically correcting it.

A new variable or mechanism is kept only if evidence shows that it improves the
model meaningfully.

No unnecessary model complexity is introduced without evidence.


---

## 30. V0 complexity

V0 uses the agreed minimal variables and mechanisms.

No additional terrain-shape metrics are required.

No technicality score is invented from GPX elevation noise.

No explicit physiological differential-equation model is required.

No unnecessary machine-learning complexity is introduced.

Additional variables or mechanisms may be introduced later as controlled
experiments.


---

## 31. Execution rules

Expensive processing runs only when explicitly started by the user.

The intended operational sequence is:

1. upload FIT files;
2. build the historical learning corpus;
3. fit the macro model;
4. build the micro historical analogue library;
5. upload the future GPX;
6. normalize the GPX;
7. enter aid stations when applicable;
8. run the simulation;
9. display and export results.

No stage should be unnecessarily executed twice for unchanged inputs.

Learning must not be repeated merely because the user interacts with displayed
results or downloads an output file.

Streamlit reruns must therefore preserve expensive results appropriately.


---

## 32. Scientific validation

The user decides whether a model is valid.

The application must expose the evidence needed for evaluation.

The system must not automatically declare a model valid or invalid.

Model development should proceed from simple representations toward additional
complexity only when evidence supports it.

When a new variable or mechanism is introduced, its effect should be compared
against the previous version under controlled conditions.


---

## 33. Development diagnostics versus independent validation

Model-development diagnostics and final independent validation are different
activities.

During development, the historical FIT corpus may be used to understand model
behaviour, diagnose failure modes, and compare candidate mechanisms.

After a candidate model has been selected, an independent validation phase may
be introduced using races that were not used during model development.

This independent validation phase is separate from model development.


---

## 34. V0 non-goals

V0 does not attempt to:

- predict individual physiological variables as primary outputs;
- build a full physiological differential-equation system;
- automatically calibrate macro and micro models against each other;
- invent technicality from noisy GPX elevation data;
- optimize the number of analogues;
- introduce complex terrain feature engineering without evidence;
- eliminate historical stationary transitions without evidence;
- automatically tune GPX granularity to match published race statistics;
- hide model instability through unreported corrections.

The purpose of V0 is to produce an interpretable macro/micro time prediction
and comparison that can be scientifically assessed.


---

## 35. Current V0 configuration summary

Current spatial configuration:

    LEARNING_STEP_M = 1.0
    GPX_SEGMENT_LENGTH_M = 100.0

Therefore:

    FIT learning density:
        every 1 m

    Historical transition horizon:
        next 100 m

    GPX normalization:
        every 100 m

    Micro target:
        observed next-100 m time

    Simulation:
        100 m per step

Current micro model:

    7 state variables
    standardized Euclidean distance
    2 nearest historical analogues
    inverse-distance interpolation

Current macro model:

    cumulative time model
    distance + cumulative ascent + cumulative descent
    linear + quadratic + interaction terms
    M(0,0,0) = 0
    negative segment increments constrained to zero in the simulator
    raw negative increments retained for diagnostics

Current GPX terrain methodology:

    raw GPX
        ->
    horizontal cumulative distance
        ->
    normalized 100 m boundaries
        ->
    linear interpolation of elevation only
        ->
    ascent/descent calculated from normalized elevation
        ->
    cumulative ascent/descent

Macro and micro remain independent.
