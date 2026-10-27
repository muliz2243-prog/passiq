// ============================================================
// PassIQ — model.rs
// Mathematical inference module.
// Loads model_weights.json, validates dimensions, and runs
// the standardize → dot-product → sigmoid pipeline.
// No UI dependencies. No side effects beyond file I/O on load.
// ============================================================

use serde::Deserialize;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

// Raw JSON structure — field names must match model_weights.json exactly.
#[derive(Deserialize)]
struct RawWeights {
    beta_distance:  f64,
    beta_angle:     f64,
    beta_pressure:  f64,
    intercept:      f64,
    training_mean:  Vec<f64>,
    training_scale: Vec<f64>,
}

// Flat, validated model coefficients ready for inference.
// Each field is named after what it represents — no index arithmetic at call sites.
pub struct PassModel {
    beta_distance:  f64,
    beta_angle:     f64,
    beta_pressure:  f64,
    intercept:      f64,
    mean_distance:  f64,
    mean_angle:     f64,
    mean_pressure:  f64,
    scale_distance: f64,
    scale_angle:    f64,
    scale_pressure: f64,
}

// Input for one inference call. All three features are required;
// the caller is responsible for computing them before calling predict().
pub struct PassInput {
    pub distance_m: f64,  // Euclidean pass length in meters
    pub angle_deg:  f64,  // Deviation from goal direction in degrees (0 = straight at goal)
    pub pressure:   f64,  // Normalized defensive pressure, 0.0–1.0
}

// All failure modes are named variants so the UI can display actionable messages.
pub enum LoadError {
    FileNotFound { path: PathBuf },
    ReadFailed   { path: PathBuf, detail: String },
    ParseFailed  { detail: String },
    BadDimensions,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::FileNotFound { path } => {
                write!(f, "File not found: {}", path.display())
            }
            LoadError::ReadFailed { path, detail } => {
                write!(f, "Cannot read {}: {detail}", path.display())
            }
            LoadError::ParseFailed { detail } => write!(f, "JSON parse error: {detail}"),
            LoadError::BadDimensions           => write!(
                f,
                "training_mean and training_scale must each have exactly 3 elements"
            ),
        }
    }
}

impl PassModel {
    // Reads the JSON file, deserializes, validates dimensions, and returns a PassModel.
    // Returns LoadError without panicking so the app can display the error in the UI.
    // Missing files are reported separately so callers can try the next candidate path.
    pub fn load(path: &Path) -> Result<PassModel, LoadError> {
        let text: String = fs::read_to_string(path).map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => LoadError::FileNotFound {
                path: path.to_path_buf(),
            },
            _ => LoadError::ReadFailed {
                path: path.to_path_buf(),
                detail: error.to_string(),
            },
        })?;

        let raw: RawWeights =
            serde_json::from_str(&text).map_err(|e| LoadError::ParseFailed {
                detail: e.to_string(),
            })?;

        if raw.training_mean.len() != 3 || raw.training_scale.len() != 3 {
            return Err(LoadError::BadDimensions);
        }

        Ok(PassModel {
            beta_distance:  raw.beta_distance,
            beta_angle:     raw.beta_angle,
            beta_pressure:  raw.beta_pressure,
            intercept:      raw.intercept,
            mean_distance:  raw.training_mean[0],
            mean_angle:     raw.training_mean[1],
            mean_pressure:  raw.training_mean[2],
            scale_distance: raw.training_scale[0],
            scale_angle:    raw.training_scale[1],
            scale_pressure: raw.training_scale[2],
        })
    }

    // Standardizes each feature with the training scaler, computes the logit,
    // applies the sigmoid, and returns xP in [0.0, 1.0].
    pub fn predict(&self, input: &PassInput) -> f64 {
        let z_dist:     f64 = (input.distance_m - self.mean_distance)  / self.scale_distance;
        let z_angle:    f64 = (input.angle_deg  - self.mean_angle)     / self.scale_angle;
        let z_pressure: f64 = (input.pressure   - self.mean_pressure)  / self.scale_pressure;

        let logit: f64 = self.intercept
            + self.beta_distance * z_dist
            + self.beta_angle    * z_angle
            + self.beta_pressure * z_pressure;

        sigmoid(logit)
    }
}

// Leaf: logistic sigmoid function. Clamps to [0, 1] by definition.
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

// ---- Domain geometry constants ----------------------------------------------
// Shared by app.rs (pitch drawing) and report.rs (batch analysis).
// Source of truth: StatsBomb coordinate system, 120 × 80 metres.

pub const PITCH_LENGTH_M: f32 = 120.0;
pub const PITCH_WIDTH_M:  f32 = 80.0;
pub const GOAL_X_M:       f32 = 120.0; // attacking goal centre X
pub const GOAL_Y_M:       f32 = 40.0;  // attacking goal centre Y

// ---- Domain geometry helpers ------------------------------------------------

// Leaf: Euclidean length of the pass vector in metres.
pub fn euclidean_distance(
    start_x: f32, start_y: f32, end_x: f32, end_y: f32,
) -> f32 {
    let dx: f32 = end_x - start_x;
    let dy: f32 = end_y - start_y;
    (dx * dx + dy * dy).sqrt()
}

// Leaf: angle in degrees between the pass vector and the vector from the pass
// origin to the attacking goal centre (GOAL_X_M, GOAL_Y_M).
// 0° = pass aimed straight at goal. Returns 0.0 on degenerate inputs.
pub fn angle_to_goal_degrees(
    start_x: f32, start_y: f32, end_x: f32, end_y: f32,
) -> f32 {
    let pass_dx: f32  = end_x - start_x;
    let pass_dy: f32  = end_y - start_y;
    let pass_len: f32 = (pass_dx * pass_dx + pass_dy * pass_dy).sqrt();

    let goal_dx: f32  = GOAL_X_M - start_x;
    let goal_dy: f32  = GOAL_Y_M - start_y;
    let goal_len: f32 = (goal_dx * goal_dx + goal_dy * goal_dy).sqrt();

    if pass_len < 0.001 || goal_len < 0.001 {
        return 0.0;
    }

    let dot: f32     = pass_dx * goal_dx + pass_dy * goal_dy;
    let cosine: f32  = (dot / (pass_len * goal_len)).clamp(-1.0, 1.0);
    cosine.acos().to_degrees()
}
