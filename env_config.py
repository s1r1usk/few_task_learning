"""
env_config.py

Shared HighwayEnv configuration for all scripts.

Observation: [5 vehicles × 7 features] = 35-dim
  features: x, y, vx, vy, cos_heading, sin_heading, presence
  relative coordinates, normalized

Trajectory: controlled vehicle state only → [T, 7]
  T = N_STEPS = 45 steps

Initial state (s0): full 5-vehicle observation flattened → [35]
  This gives the model context about surrounding traffic,
  matching the paper's use of environment context in s0.

Training scenarios: highway, merge, intersection
New concept (few-shot): roundabout  ← never seen during training
"""

N_STEPS    = 45     # trajectory length (fixed, pad/truncate to this)
OBS_DIM    = 7      # per-vehicle feature dim
N_VEHICLES = 5      # vehicles in observation
S0_DIM     = N_VEHICLES * OBS_DIM   # 35 — initial state dim
TRAJ_DIM   = OBS_DIM                # 7 — per-step trajectory dim

TRAINING_SCENARIOS = ["highway", "merge", "intersection"]
NEW_SCENARIO       = "roundabout"
ALL_SCENARIOS      = TRAINING_SCENARIOS + [NEW_SCENARIO]

SCENARIO_TO_IDX = {s: i for i, s in enumerate(TRAINING_SCENARIOS)}

ENV_IDS = {
    "highway":      "highway-v0",
    "merge":        "merge-v0",
    "roundabout":   "roundabout-v0",
    "intersection": "intersection-v0",
}

def make_env_config(scenario):
    """Return HighwayEnv config dict for a given scenario."""
    base = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": N_VEHICLES,
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h", "presence"],
            "normalize": True,
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
        "simulation_frequency": 15,
        "policy_frequency": 3,
        "duration": 15,
    }
    if scenario == "highway":
        base.update({"vehicles_count": 15, "lanes_count": 4})
    elif scenario == "merge":
        base.update({"vehicles_count": 10})
    elif scenario == "roundabout":
        base.update({"vehicles_count": 6})
    elif scenario == "intersection":
        base.update({"vehicles_count": 8})
    return base
