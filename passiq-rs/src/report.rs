// ============================================================
// PassIQ — report.rs
// Batch analysis module: loads one StatsBomb events/{match_id}.json,
// runs xP inference on every pass, and produces a per-player ranking.
//
// Style: LOTA-F — explicit, single-direction data flow.
// Driver: load_and_analyze_match() calls stages in order.
// No UI dependencies.
// ============================================================

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::model::{angle_to_goal_degrees, euclidean_distance, PassInput, PassModel};
use crate::paths;

// ---- Constants --------------------------------------------------------------

const EVENT_TYPE_PASS: &str = "Pass";

// ---- Data structures --------------------------------------------------------

// One validated pass event extracted from a StatsBomb JSON file.
struct PassRecord {
    player:         String,
    start_x:        f32,
    start_y:        f32,
    end_x:          f32,
    end_y:          f32,
    under_pressure: bool,
    outcome:        i32, // 1 = completed, 0 = failed
}

// One pass after inference: geometry, outcome and model score.
struct AnalyzedPass {
    player:         String,
    start_x:        f32,
    start_y:        f32,
    end_x:          f32,
    end_y:          f32,
    under_pressure: bool,
    completed:      bool,
    xp:             f32,
}

// One pass of a player, ready to be drawn on the pitch and loaded into the
// sliders. Keeping every pass lets the UI show a whole match for one player.
pub struct PlayerPass {
    pub start_x:        f32,
    pub start_y:        f32,
    pub end_x:          f32,
    pub end_y:          f32,
    pub under_pressure: bool,
    pub completed:      bool,
    pub xp:             f32,
}

// Running totals while ranking players; one per player name.
#[derive(Default)]
struct PlayerAccumulator {
    total_passes: u32,
    xp_sum:       f32,
    completed:    u32,
    passes:       Vec<PlayerPass>,
}

// Aggregated per-player statistics, sorted by avg_xp descending.
pub struct PlayerStat {
    pub rank:         u32,
    pub player:       String,
    pub total_passes: u32,
    pub avg_xp:       f32,
    pub completed:    u32,
    pub passes:       Vec<PlayerPass>,
}

// Output of one full match analysis.
pub struct MatchReport {
    pub match_path:   String,
    pub total_passes: u32,
    pub player_count: u32,
    pub rankings:     Vec<PlayerStat>,
}

// Top-level state: no match loaded, a loaded report, or a load error.
pub enum MatchLoadState {
    Empty,
    Loaded(MatchReport),
    Error(String),
}

// ---- Pure functions, in the order the driver calls them --------------------

fn load_events_json(path: &Path) -> Result<Vec<Value>, String> {
    let text: String = fs::read_to_string(path)
        .map_err(|e| format!("Cannot read '{}': {}", path.display(), e))?;

    let root: Value = serde_json::from_str(&text)
        .map_err(|e| format!("Invalid JSON in '{}': {}", path.display(), e))?;

    root.as_array()
        .ok_or_else(|| {
            format!("'{}' must contain a JSON array at the top level.", path.display())
        })
        .map(|arr| arr.to_vec())
}

fn extract_pass_records(events: Vec<Value>) -> Vec<PassRecord> {
    let mut records: Vec<PassRecord> = Vec::new();

    for event in events {
        // Skip non-pass events
        let event_type: &str = event
            .get("type")
            .and_then(|t| t.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("");

        if event_type != EVENT_TYPE_PASS {
            continue;
        }

        let player: String = event
            .get("player")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("Unknown Player")
            .to_owned();

        // Both location arrays are required; skip if absent
        let location    = event.get("location").and_then(|v| v.as_array());
        let pass_obj    = event.get("pass").and_then(|v| v.as_object());

        if location.is_none() || pass_obj.is_none() {
            continue;
        }

        let loc:  &Vec<Value> = location.unwrap();
        let pass: &serde_json::Map<String, Value> = pass_obj.unwrap();

        let end_location = pass.get("end_location").and_then(|v| v.as_array());
        if end_location.is_none() || loc.len() < 2 {
            continue;
        }

        let end_loc: &Vec<Value> = end_location.unwrap();
        if end_loc.len() < 2 {
            continue;
        }

        let start_x: f32 = loc[0].as_f64().unwrap_or(0.0) as f32;
        let start_y: f32 = loc[1].as_f64().unwrap_or(0.0) as f32;
        let end_x:   f32 = end_loc[0].as_f64().unwrap_or(0.0) as f32;
        let end_y:   f32 = end_loc[1].as_f64().unwrap_or(0.0) as f32;

        // StatsBomb: "under_pressure" key is absent when False
        let under_pressure: bool = event
            .get("under_pressure")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // StatsBomb: "outcome" key is absent when the pass is completed
        let outcome: i32 = if pass.get("outcome").is_some() { 0 } else { 1 };

        records.push(PassRecord {
            player,
            start_x,
            start_y,
            end_x,
            end_y,
            under_pressure,
            outcome,
        });
    }

    records
}

fn analyze_passes(records: &[PassRecord], model: &PassModel) -> Vec<AnalyzedPass> {
    let mut analyzed: Vec<AnalyzedPass> = Vec::with_capacity(records.len());

    for record in records {
        let distance_m: f32 = euclidean_distance(
            record.start_x, record.start_y, record.end_x, record.end_y,
        );
        let angle_deg: f32 = angle_to_goal_degrees(
            record.start_x, record.start_y, record.end_x, record.end_y,
        );
        // Binary pressure mapping: under pressure = 1.0, free = 0.0
        let pressure: f32 = if record.under_pressure { 1.0 } else { 0.0 };

        let input: PassInput = PassInput {
            distance_m: distance_m as f64,
            angle_deg:  angle_deg as f64,
            pressure:   pressure as f64,
        };

        let xp: f32 = model.predict(&input) as f32;

        analyzed.push(AnalyzedPass {
            player:         record.player.clone(),
            start_x:        record.start_x,
            start_y:        record.start_y,
            end_x:          record.end_x,
            end_y:          record.end_y,
            under_pressure: record.under_pressure,
            completed:      record.outcome == 1,
            xp,
        });
    }

    analyzed
}

fn rank_players(passes: &[AnalyzedPass]) -> Vec<PlayerStat> {
    // BTreeMap keeps keys ordered — intermediate iteration is deterministic.
    let mut acc: BTreeMap<String, PlayerAccumulator> = BTreeMap::new();

    for pass in passes {
        let entry: &mut PlayerAccumulator = acc
            .entry(pass.player.clone())
            .or_default();
        entry.total_passes += 1;
        entry.xp_sum += pass.xp;
        if pass.completed {
            entry.completed += 1;
        }
        entry.passes.push(PlayerPass {
            start_x:        pass.start_x,
            start_y:        pass.start_y,
            end_x:          pass.end_x,
            end_y:          pass.end_y,
            under_pressure: pass.under_pressure,
            completed:      pass.completed,
            xp:             pass.xp,
        });
    }

    // Collect into a Vec so we can sort by avg_xp (BTreeMap sorts by key only)
    let mut rows: Vec<(String, PlayerAccumulator)> = acc.into_iter().collect();

    // Primary sort: avg_xp descending. Secondary: player name ascending (stable tie-break).
    rows.sort_by(|a, b| {
        let avg_a: f32 = a.1.xp_sum / a.1.total_passes as f32;
        let avg_b: f32 = b.1.xp_sum / b.1.total_passes as f32;
        avg_b
            .partial_cmp(&avg_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    rows.into_iter()
        .enumerate()
        .map(|(i, (player, entry))| PlayerStat {
            rank:         (i + 1) as u32,
            player,
            total_passes: entry.total_passes,
            avg_xp:       entry.xp_sum / entry.total_passes as f32,
            completed:    entry.completed,
            passes:       entry.passes,
        })
        .collect()
}

// ---- Driver -----------------------------------------------------------------

// Loads one StatsBomb events/{match_id}.json, runs the full pipeline, and
// returns a MatchReport. Returns Err with a descriptive message on any failure.
// Relative paths are resolved against the project root, not the process directory.
pub fn load_and_analyze_match(path: &str, model: &PassModel) -> Result<MatchReport, String> {
    let resolved: PathBuf = paths::resolve_existing(path)
        .ok_or_else(|| format!("File not found: '{}'", path))?;

    let events: Vec<Value>          = load_events_json(resolved.as_path())?;
    let records: Vec<PassRecord>    = extract_pass_records(events);

    if records.is_empty() {
        return Err(format!(
            "No valid pass events found in '{}'.",
            resolved.display()
        ));
    }

    let passes: Vec<AnalyzedPass>   = analyze_passes(&records, model);
    let rankings:  Vec<PlayerStat>  = rank_players(&passes);

    let player_count: u32 = rankings.len() as u32;
    let total_passes: u32 = passes.len() as u32;
    Ok(MatchReport {
        match_path:   resolved.display().to_string(),
        total_passes,
        player_count,
        rankings,
    })
}
