# ============================================================
# PassIQ — train_model.py
# Phase 1: Trains an xP (Expected Pass) logistic regression model
# on StatsBomb-format pass events and exports the learned
# coefficients to model_weights.json for Rust inference.
#
# Style: LOTA-F — explicit, single-direction data flow.
# ============================================================

# 1. Imports
import glob
import json
import math
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# 2. True constants — fixed domain values that never change at runtime
FIELD_LENGTH_METERS: float = 120.0   # StatsBomb pitch X axis
FIELD_WIDTH_METERS:  float = 80.0    # StatsBomb pitch Y axis
GOAL_X:              float = 120.0   # attacking goal center X
GOAL_Y:              float = 40.0    # attacking goal center Y
EVENT_TYPE_PASS:     str   = "Pass"  # StatsBomb event type name for passes


# 3. Data structures
@dataclass
class PassRecord:
    """One validated pass event extracted from a StatsBomb JSON file."""
    pass_id:        str
    start_x:        float   # meters, 0.0–120.0
    start_y:        float   # meters, 0.0–80.0
    end_x:          float   # meters, 0.0–120.0
    end_y:          float   # meters, 0.0–80.0
    under_pressure: bool
    outcome:        int     # 1 = completed, 0 = incomplete/failed


@dataclass
class ModelWeights:
    """Coefficients and scaler parameters exported to model_weights.json."""
    beta_distance:       float
    beta_angle:          float
    beta_pressure:       float
    intercept:           float
    training_mean:       list[float]   # [mu_dist, mu_angle, mu_pressure]
    training_scale:      list[float]   # [sigma_dist, sigma_angle, sigma_pressure]


# 4. Pure functions, in the order the driver calls them
# ----------------------------------------------------------

def load_events_json(path: str) -> list[dict]:
    """
    Reads one StatsBomb events/{match_id}.json file and returns the raw event list.
    Exits the process with a descriptive message on missing file or malformed JSON.
    """
    try:
        with open(path, encoding="utf-8") as f:
            content = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: file not found — '{path}'")
        sys.exit(1)
    except json.JSONDecodeError as err:
        print(f"ERROR: invalid JSON in '{path}'. Detail: {err}")
        sys.exit(1)

    if not isinstance(content, list):
        print(
            f"ERROR: expected a JSON array in '{path}', "
            f"got {type(content).__name__}."
        )
        sys.exit(1)

    return content


def extract_valid_passes(events: list[dict]) -> list[PassRecord]:
    """
    Filters Pass events from the raw event list and builds a PassRecord for each.
    Discards events with missing or malformed coordinate fields.
    Reports counts to stdout so the pipeline is auditable.
    """
    valid_passes: list[PassRecord] = []
    skipped_type: int   = 0
    skipped_fields: int = 0

    for event in events:
        event_type: str = event.get("type", {}).get("name", "")
        if event_type != EVENT_TYPE_PASS:
            skipped_type += 1
            continue

        pass_id: str   = event.get("id", "unknown")
        location       = event.get("location")
        pass_data: dict = event.get("pass", {})
        end_location   = pass_data.get("end_location")

        if (
            location is None
            or end_location is None
            or len(location) < 2
            or len(end_location) < 2
        ):
            print(f"  WARNING: pass '{pass_id}' dropped — missing coordinates.")
            skipped_fields += 1
            continue

        start_x: float = float(location[0])
        start_y: float = float(location[1])
        end_x:   float = float(end_location[0])
        end_y:   float = float(end_location[1])

        # StatsBomb: "under_pressure" key is absent when False
        under_pressure: bool = bool(event.get("under_pressure", False))

        # StatsBomb: "outcome" key is absent when the pass is completed
        raw_outcome = pass_data.get("outcome")
        outcome: int = 0 if raw_outcome is not None else 1

        valid_passes.append(PassRecord(
            pass_id=pass_id,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            under_pressure=under_pressure,
            outcome=outcome,
        ))

    print(
        f"Extraction: {len(valid_passes)} valid passes | "
        f"{skipped_type} non-pass events skipped | "
        f"{skipped_fields} passes dropped (missing fields)."
    )

    if len(valid_passes) == 0:
        print("ERROR: no valid passes found. Cannot train.")
        sys.exit(1)

    return valid_passes


def _euclidean_distance(
    start_x: float, start_y: float, end_x: float, end_y: float
) -> float:
    """Leaf: Euclidean length of the pass vector in meters."""
    dx: float = end_x - start_x
    dy: float = end_y - start_y
    return math.sqrt(dx * dx + dy * dy)


def _angle_to_goal_degrees(
    start_x: float, start_y: float, end_x: float, end_y: float
) -> float:
    """
    Leaf: angle in degrees between the pass vector and the vector from the
    pass origin to the attacking goal center (GOAL_X, GOAL_Y).
    0° = pass aimed directly at goal. Returns 0.0 on zero-length vectors.
    """
    pass_dx: float  = end_x - start_x
    pass_dy: float  = end_y - start_y
    pass_len: float = math.sqrt(pass_dx * pass_dx + pass_dy * pass_dy)

    goal_dx: float  = GOAL_X - start_x
    goal_dy: float  = GOAL_Y - start_y
    goal_len: float = math.sqrt(goal_dx * goal_dx + goal_dy * goal_dy)

    if pass_len == 0.0 or goal_len == 0.0:
        return 0.0

    dot: float     = pass_dx * goal_dx + pass_dy * goal_dy
    cosine: float  = dot / (pass_len * goal_len)
    clamped: float = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(clamped))


def build_feature_matrix(passes: list[PassRecord]) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms a list of PassRecord into two numpy arrays:
      X: shape (n, 3) — columns: [distance_m, angle_deg, pressure_binary]
      y: shape (n,)   — labels:  [0 or 1]

    No normalization here; that happens inside train_logistic_model with StandardScaler.
    """
    rows: list[list[float]] = []
    labels: list[int]       = []

    for p in passes:
        distance: float = _euclidean_distance(p.start_x, p.start_y, p.end_x, p.end_y)
        angle: float    = _angle_to_goal_degrees(p.start_x, p.start_y, p.end_x, p.end_y)
        pressure: float = 1.0 if p.under_pressure else 0.0

        rows.append([distance, angle, pressure])
        labels.append(p.outcome)

    X: np.ndarray = np.array(rows,   dtype=np.float64)
    y: np.ndarray = np.array(labels, dtype=np.int32)

    n_positive: int = int(np.sum(y))
    n_negative: int = len(y) - n_positive
    print(f"Features: {len(passes)} passes | {n_positive} completed | {n_negative} failed.")

    if n_positive == 0:
        print("ERROR: no positive examples (completed passes). Cannot train.")
        sys.exit(1)

    if n_negative == 0:
        print("ERROR: no negative examples (failed passes). Cannot train.")
        sys.exit(1)

    return X, y


def train_logistic_model(
    X: np.ndarray, y: np.ndarray
) -> tuple[LogisticRegression, StandardScaler]:
    """
    Standardizes X and fits a logistic regression classifier.
    Returns both the fitted model and the scaler; both are needed to
    reconstruct the full inference formula in Rust.

    Hyperparameter notes:
      solver='lbfgs'        — default for binary logistic regression, L2 penalty.
      max_iter=1000         — football features can be correlated; more iterations needed.
      random_state=42       — deterministic output for the same training data.
      class_weight='balanced' — compensates the natural imbalance (more completed passes).
    """
    scaler: StandardScaler = StandardScaler()
    X_scaled: np.ndarray   = scaler.fit_transform(X)

    model: LogisticRegression = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_scaled, y)

    print(f"Training: converged in {model.n_iter_[0]} iterations.")
    return model, scaler


def extract_weights(
    model: LogisticRegression, scaler: StandardScaler
) -> ModelWeights:
    """
    Reads raw sklearn coefficients and scaler parameters, returns a ModelWeights
    struct. The three betas correspond to the feature order [distance, angle, pressure].
    """
    coefs: np.ndarray  = model.coef_[0]         # shape (3,)
    bias:  float       = float(model.intercept_[0])

    return ModelWeights(
        beta_distance=float(coefs[0]),
        beta_angle=float(coefs[1]),
        beta_pressure=float(coefs[2]),
        intercept=bias,
        training_mean=scaler.mean_.tolist(),
        training_scale=scaler.scale_.tolist(),
    )


def export_weights_json(weights: ModelWeights, output_path: str) -> None:
    """
    Serializes ModelWeights to a flat JSON file for the Rust binary to consume.
    Flat structure — no nesting — keeps Rust deserialization trivial.
    """
    payload: dict = {
        "beta_distance":   weights.beta_distance,
        "beta_angle":      weights.beta_angle,
        "beta_pressure":   weights.beta_pressure,
        "intercept":       weights.intercept,
        "training_mean":   weights.training_mean,
        "training_scale":  weights.training_scale,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Weights exported to '{output_path}':")
    print(json.dumps(payload, indent=2))



# 5. Driver — orchestrates stages in order, contains no business logic
# --------------------------------------------------------------------

def collect_real_paths(directory: str) -> list[str]:
    """
    Returns a sorted list of .json file paths found under `directory`.
    Sorted for deterministic order across runs (LOTA-F Rule 13).
    Returns an empty list when the directory does not exist or has no JSON files.
    """
    pattern: str     = directory.rstrip("/\\") + "/*.json"
    found: list[str] = sorted(glob.glob(pattern))
    return found


def load_multiple_files(paths: list[str]) -> list[dict]:
    """
    Calls load_events_json on each path and concatenates all event lists.
    Processes paths in order; stops on the first file error.
    Returns a single flat list of raw event dicts.
    """
    all_events: list[dict] = []

    for path in paths:
        file_events: list[dict] = load_events_json(path)
        print(f"  Loaded '{path}': {len(file_events)} events.")
        all_events = all_events + file_events

    return all_events


def main() -> None:
    output_path: str = "model_weights.json"

    print("=== PassIQ — Phase 1: xP Model Training ===\n")

    real_paths: list[str]  = collect_real_paths("data/real")
    input_paths: list[str] = real_paths if len(real_paths) > 0 else ["data/sample_match.json"]

    if len(real_paths) > 0:
        print(f"Using {len(real_paths)} real StatsBomb file(s): {input_paths}")
    else:
        print("No real data found — falling back to synthetic sample.")

    raw_events: list[dict]          = load_multiple_files(input_paths)
    passes: list[PassRecord]        = extract_valid_passes(raw_events)
    X_y                             = build_feature_matrix(passes)
    X: np.ndarray                   = X_y[0]
    y: np.ndarray                   = X_y[1]
    model, scaler                   = train_logistic_model(X, y)
    weights: ModelWeights           = extract_weights(model, scaler)
    export_weights_json(weights, output_path)

    print("\n=== Phase 1 complete ===")


if __name__ == "__main__":
    main()