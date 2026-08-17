# Trail Running Simulator V0 Specification

## 1. Objective

The simulator predicts elapsed time for every normalized 50 m segment of a future GPX race.

The primary outputs are:

- predicted time for each 50 m segment;
- predicted cumulative time;
- predicted arrival time at each aid station;
- predicted total race time.

The simulator does not need to predict HR, cadence, power, cardiovascular debt, mechanical debt, or neuromuscular debt as primary outputs.

---

## 2. Two independent prediction models

The simulator contains two independent models.

### 2.1 Macro model

The macro model learns the relationship between:

- distance from start;
- cumulative ascent;
- cumulative descent;

and cumulative elapsed race time.

Conceptually:

T_macro = M(distance, cumulative_ascent, cumulative_descent)

The macro model provides an independent macro-scale prediction.

It does not modify the micro model.

---

### 2.2 Micro model

The micro model predicts the elapsed time for the next 50 m.

For a prediction position x, the state vector is:

1. distance from start;
2. cumulative ascent from start;
3. cumulative descent from start;
4. cumulative elapsed predicted time;
5. ascent of the coming 50 m;
6. descent of the coming 50 m;
7. average grade of the coming 50 m.

Conceptually:

delta_t_50 = F(state)

The historical FIT corpus provides observations of this relationship.

The micro model is empirical / analog based.

---

## 3. Historical FIT learning

All uploaded FIT files are processed together.

For every FIT:

- reconstruct the historical trajectory;
- use a rolling 1 m starting position;
- for every position d where d + 50 m exists, create one historical transition.

Each historical transition contains:

- distance from start;
- cumulative ascent;
- cumulative descent;
- elapsed time from start;
- ascent over the next 50 m;
- descent over the next 50 m;
- average grade over the next 50 m;
- actual elapsed time over the next 50 m.

The target is:

actual_segment_time_s =
time_at_d_plus_50m - time_at_d

The learning window and prediction window are both 50 m.

---

## 4. Micro analog search

A future GPX prediction state is compared with all historical states.

"Closest" means the historical state whose complete multivariable state is most similar.

It does NOT mean geographically closest distance.

For V0, the state distance is standardized Euclidean distance.

Each variable is normalized using its historical standard deviation.

For historical state H and query state Q:

D(H,Q) =
sqrt(
    sum(
        ((Q_j - H_j) / sigma_j)^2
    )
)

The two historical states with the smallest distance are selected.

---

## 5. Micro prediction

The two closest historical transitions provide two observed next-50 m times.

Let:

- D1 = distance to closest state;
- D2 = distance to second closest state;
- t1 = observed next-50 m time for closest state;
- t2 = observed next-50 m time for second closest state.

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

If the closest distance is effectively zero, use the observed time of the closest state directly.

The number of analogues is fixed at two for V0.

---

## 6. Recursive simulation

The simulator proceeds through the GPX in 50 m segments.

For each segment:

1. construct the current prediction state;
2. calculate the macro prediction;
3. calculate the micro prediction;
4. store both predictions;
5. update cumulative macro time;
6. update cumulative micro time;
7. move to the next 50 m.

No macro-to-micro correction is performed.

No micro-to-macro correction is performed.

No clipping or blending is performed.

Macro and micro remain independent.

---

## 7. Aid stations

Each aid station may contain:

- aid station name;
- distance from race start;
- expected stop duration in minutes.

Aid station stop duration represents elapsed clock time spent stopped.

The stop duration is added independently to both macro and micro cumulative clocks.

For both models:

cumulative_time_after_stop =
cumulative_time_before_stop + stop_duration

Aid-station stop time is not treated as running time.

---

## 8. GPX normalization

The future GPX is normalized into 50 m segments.

The end of each segment is used as the value of:

distance_from_start_m

Example:

distance_from_start_m = 5000

represents the segment from 4950 m to 5000 m.

Terrain values must use the same definitions in historical FIT learning and future GPX prediction.

---

## 9. Main output

One row represents one normalized 50 m segment.

Required columns:

- distance_from_start_m
- ascent_m
- descent_m
- cumulative_ascent_m
- cumulative_descent_m
- grade_pct
- macro_predicted_time_s
- micro_predicted_time_s
- segment_difference_s
- macro_predicted_cumulative_time_s
- micro_predicted_cumulative_time_s
- cumulative_difference_s

Aid station columns:

- aid_station_name
- aid_station_stop_min

Human-readable time columns may additionally be provided.

---

## 10. Macro / micro comparison

The output must show macro and micro predictions side by side.

For each segment:

segment_difference_s =
micro_predicted_time_s - macro_predicted_time_s

For cumulative time:

cumulative_difference_s =
micro_predicted_cumulative_time_s
- macro_predicted_cumulative_time_s

These differences are diagnostic only.

They must never be used to automatically modify either model.

The user decides whether the model is valid.

---

## 11. Model development methodology

The model is developed incrementally.

For each experiment:

- use the same complete FIT corpus;
- start with the simplest model;
- add exactly one variable or mechanism;
- rebuild the model using the same FIT corpus;
- compare the resulting predictions.

A new variable or mechanism is kept only if the evidence shows that it improves the model meaningfully.

No unnecessary model complexity is introduced without evidence.

---

## 12. V0 complexity

V0 starts with the agreed minimal variables.

No additional terrain-shape metrics are required.

No technicality score is invented from GPX elevation noise.

No explicit physiological differential-equation model is required.

Additional variables or mechanisms can be introduced later as controlled experiments.

---

## 13. Macro model independence

The macro model and micro model are independent.

The macro model is not a correction mechanism for the micro model.

The micro model is not a correction mechanism for the macro model.

Both predictions are presented to the user for comparison.

---

## 14. Execution rules

The expensive processing pipeline runs only when the user explicitly starts the analysis.

The intended sequence is:

1. upload FIT files;
2. upload GPX;
3. enter aid stations;
4. start analysis;
5. prepare historical FIT data;
6. learn macro model;
7. build micro historical state library;
8. normalize GPX;
9. simulate the race;
10. display and export the results.

No stage should be unnecessarily executed twice for unchanged inputs.

Learning must not be repeated merely because the user interacts with the results.

---

## 15. Scientific validation

The user decides whether a model is valid.

The application must expose the evidence needed for evaluation.

The system must not automatically declare a model valid or invalid.

Model development should proceed from simple to more granular representations.

When a new variable or mechanism is introduced, its effect must be compared against the previous version using the same historical FIT corpus.

---

## 16. Future validation phase

After a candidate model has been selected, an independent validation phase may be introduced using races that were not used during model development.

This is separate from the model-development process.

---

## 17. V0 non-goals

V0 does not attempt to:

- predict individual physiological variables;
- build a full physiological differential-equation system;
- automatically calibrate macro and micro models against each other;
- invent technicality from noisy elevation data;
- optimize the number of analogues;
- introduce complex terrain feature engineering without evidence.

The purpose of V0 is to produce an interpretable macro/micro time prediction and comparison that can be scientifically assessed.
