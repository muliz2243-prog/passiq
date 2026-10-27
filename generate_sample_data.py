# ============================================================
# PassIQ — generate_sample_data.py
# Generates a synthetic StatsBomb-format JSON file (one file
# per match, one event per object) so that train_model.py can
# run without downloading real data from StatsBomb Open Data.
#
# Style: LOTA-F — explicit, single-direction data flow.
# ============================================================

# 1. Imports
import json
import math
import random
import sys
import uuid
from dataclasses import dataclass


# 2. True constants — fixed domain values that never change at runtime
FIELD_LENGTH_METERS: float = 120.0   # StatsBomb pitch X axis
FIELD_WIDTH_METERS:  float = 80.0    # StatsBomb pitch Y axis
GOAL_X:              float = 120.0   # attacking goal center X
GOAL_Y:              float = 40.0    # attacking goal center Y

TOTAL_PASSES:        int   = 600     # number of synthetic pass events to generate
RANDOM_SEED:         int   = 2026    # fixed seed guarantees deterministic output
OUTPUT_PATH:         str   = "data/sample_match.json"


# 3. Data structures
@dataclass
class PassParams:
    """All parameters needed to build one StatsBomb Pass event."""
    start_x:       float   # meters, 0.0–120.0
    start_y:       float   # meters, 0.0–80.0
    end_x:         float   # meters, 0.0–120.0
    end_y:         float   # meters, 0.0–80.0
    under_pressure: bool
    successful:    bool    # True = completed pass (no "outcome" key in pass dict)


# 4. Pure functions, in the order the driver calls them
# ----------------------------------------------------------

def generate_start_position(rng: random.Random) -> tuple[float, float]:
    """Returns a realistic pass origin, avoiding byline extremes."""
    start_x: float = rng.uniform(5.0, 115.0)
    start_y: float = rng.uniform(5.0, 75.0)
    return start_x, start_y


def generate_end_position(
    start_x: float, start_y: float, rng: random.Random
) -> tuple[float, float]:
    """
    Returns a pass destination within a realistic distance range (3–40 m).
    Clamps the result to keep the ball inside the pitch boundary.
    """
    distance: float = rng.uniform(3.0, 40.0)
    angle_rad: float = rng.uniform(0.0, 2.0 * math.pi)

    raw_end_x: float = start_x + distance * math.cos(angle_rad)
    raw_end_y: float = start_y + distance * math.sin(angle_rad)

    end_x: float = max(1.0, min(FIELD_LENGTH_METERS - 1.0, raw_end_x))
    end_y: float = max(1.0, min(FIELD_WIDTH_METERS  - 1.0, raw_end_y))
    return end_x, end_y


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
    Leaf: angle in degrees between the pass vector and the vector
    from the pass origin to the attacking goal center.
    0° = pass aimed directly at goal. Returns 90.0 on zero-length input.
    """
    pass_dx: float = end_x - start_x
    pass_dy: float = end_y - start_y
    pass_len: float = math.sqrt(pass_dx * pass_dx + pass_dy * pass_dy)

    goal_dx: float = GOAL_X - start_x
    goal_dy: float = GOAL_Y - start_y
    goal_len: float = math.sqrt(goal_dx * goal_dx + goal_dy * goal_dy)

    if pass_len == 0.0 or goal_len == 0.0:
        return 90.0

    dot: float = pass_dx * goal_dx + pass_dy * goal_dy
    cosine: float = dot / (pass_len * goal_len)
    cosine_clamped: float = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine_clamped))


def compute_success_probability(
    start_x: float, start_y: float,
    end_x: float, end_y: float,
    under_pressure: bool,
) -> float:
    """
    Returns a synthetic but realistic success probability using a hand-crafted
    logistic formula. Encodes the same signal that the logistic model must learn:
      - shorter passes → higher probability
      - smaller goal angle → higher probability (forward pass)
      - under pressure → lower probability
    Produces ~75% success rate, consistent with elite football statistics.
    """
    distance: float      = _euclidean_distance(start_x, start_y, end_x, end_y)
    angle: float         = _angle_to_goal_degrees(start_x, start_y, end_x, end_y)
    pressure_val: float  = 1.0 if under_pressure else 0.0

    logit: float = (
        2.5
        - 0.07 * distance
        - 0.01 * angle
        - 0.80 * pressure_val
    )
    return 1.0 / (1.0 + math.exp(-logit))


def build_pass_params(rng: random.Random) -> PassParams:
    """Assembles all parameters for one synthetic pass event."""
    start_x, start_y = generate_start_position(rng)
    end_x, end_y     = generate_end_position(start_x, start_y, rng)

    # ~30% of passes occur under pressure in modern elite football
    under_pressure: bool = rng.random() < 0.30

    prob: float       = compute_success_probability(start_x, start_y, end_x, end_y, under_pressure)
    successful: bool  = rng.random() < prob

    return PassParams(
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        under_pressure=under_pressure,
        successful=successful,
    )


def build_statsbomb_event(params: PassParams, event_index: int) -> dict:
    """
    Constructs one event object following the official StatsBomb schema
    (events/{match_id}.json format).

    StatsBomb convention for pass outcome:
      completed pass  → "outcome" key is ABSENT  from the pass dict
      failed pass     → "outcome" key is PRESENT with a failure name
    """
    pass_data: dict = {
        "end_location": [round(params.end_x, 2), round(params.end_y, 2)],
        "length": round(
            _euclidean_distance(params.start_x, params.start_y, params.end_x, params.end_y), 2
        ),
        "angle": round(
            math.atan2(params.end_y - params.start_y, params.end_x - params.start_x), 4
        ),
        "height":    {"id": 1, "name": "Ground Pass"},
        "recipient": {"id": 5000 + event_index, "name": f"Player_{event_index % 11}"},
    }

    if not params.successful:
        pass_data["outcome"] = {"id": 9, "name": "Incomplete"}

    event: dict = {
        "id":       str(uuid.UUID(int=event_index, version=4)),
        "index":    event_index,
        "period":   1 if event_index <= TOTAL_PASSES // 2 else 2,
        "timestamp": f"00:{(event_index % 45):02d}:{(event_index % 60):02d}.000",
        "minute":   event_index % 90,
        "second":   event_index % 60,
        "type":     {"id": 30, "name": "Pass"},
        "possession": event_index,
        "possession_team": {"id": 217, "name": "Home Team"},
        "play_pattern":   {"id": 1,   "name": "Regular Play"},
        "team":     {"id": 217, "name": "Home Team"},
        "player":   {"id": 3000 + (event_index % 11), "name": f"Player_{event_index % 11}"},
        "location": [round(params.start_x, 2), round(params.start_y, 2)],
        "pass":     pass_data,
    }

    # StatsBomb only includes "under_pressure" when True
    if params.under_pressure:
        event["under_pressure"] = True

    return event


def generate_event_list(total_passes: int, rng: random.Random) -> list[dict]:
    """
    Generates the full event list for one synthetic match.
    Inserts a non-Pass event every 10 passes to exercise the filter in train_model.py.
    """
    events: list[dict] = []
    global_index: int = 1

    for i in range(total_passes):
        params: PassParams = build_pass_params(rng)
        events.append(build_statsbomb_event(params, global_index))
        global_index += 1

        if i % 10 == 0:
            non_pass_event: dict = {
                "id":       str(uuid.UUID(int=global_index, version=4)),
                "index":    global_index,
                "period":   1,
                "type":     {"id": 23, "name": "Ball Receipt*"},
                "location": [40.0, 40.0],
            }
            events.append(non_pass_event)
            global_index += 1

    return events


def compute_event_stats(events: list[dict]) -> dict:
    """
    Returns a summary dict with pass counts for console reporting.
    Does not modify the event list.
    """
    successful_passes: int = 0
    failed_passes: int     = 0
    other_events: int      = 0

    for event in events:
        if event.get("type", {}).get("name") != "Pass":
            other_events += 1
            continue
        outcome = event.get("pass", {}).get("outcome")
        if outcome is None:
            successful_passes += 1
        else:
            failed_passes += 1

    return {
        "total_events":     len(events),
        "successful_passes": successful_passes,
        "failed_passes":    failed_passes,
        "other_events":     other_events,
    }


def write_json_file(events: list[dict], output_path: str) -> None:
    """Writes the event list to disk as a StatsBomb-format JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
    except OSError as err:
        print(f"ERROR: could not write '{output_path}'. Detail: {err}")
        sys.exit(1)

    print(f"File written: '{output_path}'")


# 5. Driver — orchestrates stages in order, contains no business logic
# --------------------------------------------------------------------

def main() -> None:
    print("=== PassIQ — generate_sample_data.py ===\n")
    print(f"Generating {TOTAL_PASSES} synthetic pass events (seed={RANDOM_SEED})...")

    rng: random.Random = random.Random(RANDOM_SEED)

    events: list[dict] = generate_event_list(TOTAL_PASSES, rng)
    stats: dict        = compute_event_stats(events)

    print(f"  Total events generated : {stats['total_events']}")
    print(f"  Successful passes      : {stats['successful_passes']}")
    print(f"  Failed passes          : {stats['failed_passes']}")
    print(f"  Other event types      : {stats['other_events']}")

    write_json_file(events, OUTPUT_PATH)

    print("\nDone. Run next: python train_model.py")


if __name__ == "__main__":
    main()
