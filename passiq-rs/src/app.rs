// ============================================================
// PassIQ — app.rs
// UI state, inference pipeline, and egui rendering.
//
// Style: LOTA-F — explicit, single-direction data flow.
//
// Stage order inside update():
//   1. compute_inference      → InferenceResult   (pure, no UI)
//   2. show_top_panel         → xP readout with dynamic color
//   3. show_left_panel        → sliders + reload model + load match
//   4. show_right_panel       → player ranking table     ← NEW
//   5. show_central_panel     → pitch canvas + pass arrow
//      (CentralPanel must come last — fills remaining space)
// ============================================================

use egui::{Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Vec2};
use std::path::PathBuf;

use crate::model::{
    angle_to_goal_degrees, LoadError, PassInput, PassModel,
    PITCH_LENGTH_M, PITCH_WIDTH_M,
};
use crate::paths;
use crate::report::{
    load_and_analyze_match, MatchLoadState, MatchReport, PlayerPass, PlayerStat,
};

// ---- UI constants -----------------------------------------------------------

const XP_SAFE_THRESHOLD:    f32 = 0.70;
const XP_CAUTION_THRESHOLD: f32 = 0.35;

// Single source of truth for the risk palette (legend, top bar, ranking rows).
const COLOR_SAFE:    Color32 = Color32::from_rgb(72, 199, 116);
const COLOR_CAUTION: Color32 = Color32::from_rgb(255, 165, 0);
const COLOR_HIGH:    Color32 = Color32::from_rgb(230, 57, 70);

// Defenders that map to full pressure (defenders / 5.0 = 1.0).
const FULL_PRESSURE_DEFENDERS: f32 = 5.0;

// Longest pass the sliders can describe: a full-length pass of the pitch.
const MAX_PASS_DISTANCE_M: f32 = PITCH_LENGTH_M;

// Click tolerance around a drawn pass, in screen pixels.
const PASS_PICK_RADIUS_PX: f32 = 8.0;

// Real La Liga 2020/21 matches available for one-click loading.
// Paths are relative to the project root; paths::resolve_existing() turns them
// into absolute paths even when the app was started from another directory.
const QUICK_MATCHES: &[(&str, &str)] = &[
    ("Real Madrid 2–1 Barcelona  (La Liga 2020/21)", "data/real/3773497.json"),
    ("Barcelona 1–3 Real Madrid  (La Liga 2020/21)", "data/real/3773585.json"),
    ("Alavés 1–1 Barcelona       (La Liga 2020/21)", "data/real/3773386_Deportivo_Alav\u{e9}s_vs_Barcelona.json"),
    ("Synthetic sample (600 passes)",                "data/sample_match.json"),
];

// ---- Data structures --------------------------------------------------------

// All mutable slider state. Drives both live inference and pitch drawing.
pub struct SliderState {
    pub start_x:       f32, // pass origin X, pitch metres 0.0–120.0
    pub start_y:       f32, // pass origin Y, pitch metres 0.0–80.0
    pub distance_m:    f32, // pass length in metres, 0.0–MAX_PASS_DISTANCE_M
    pub direction_deg: f32, // absolute pass direction, degrees −180 to 180
    pub defenders:     f32, // defenders in path, 0.0–5.0 → mapped to pressure
}

// Risk classification used for color coding and label text.
pub enum RiskLevel {
    Safe,    // xP > 0.70 — pass likely to succeed
    Caution, // 0.35 ≤ xP ≤ 0.70 — moderate risk
    High,    // xP < 0.35 — high risk / line-breaking pass
}

// Everything needed by the UI to render one frame's inference output.
pub struct InferenceResult {
    pub xp:    f32,
    pub risk:  RiskLevel,
    pub color: Color32,
}

// Loaded model or load error. Never panics; error shown in the UI.
pub enum ModelState {
    Loaded(PassModel),
    Failed(String),
}

// Which player's passes are drawn on the pitch and which one is loaded into
// the sliders. Both indices refer to MatchReport::rankings.
#[derive(Default)]
pub struct Selection {
    pub player_index: Option<usize>,
    pub pass_index:   Option<usize>,
}

// Root application state.
pub struct PassIQApp {
    pub model_state:      ModelState,
    pub sliders:          SliderState,
    pub match_path_input: String,      // text buffer for the match JSON path
    pub match_load_state: MatchLoadState,
    pub selection:        Selection,
}

// Affine transform: StatsBomb pitch coords → egui screen pixels.
// Preserves aspect ratio and centres the pitch inside the available rect.
struct PitchTransform {
    offset_x: f32,
    offset_y: f32,
    scale:    f32,
}

// ---- Pure helper functions (no UI, no side effects) -------------------------

pub fn classify_risk(xp: f32) -> RiskLevel {
    if xp > XP_SAFE_THRESHOLD {
        RiskLevel::Safe
    } else if xp >= XP_CAUTION_THRESHOLD {
        RiskLevel::Caution
    } else {
        RiskLevel::High
    }
}

pub fn risk_to_color(risk: &RiskLevel) -> Color32 {
    match risk {
        RiskLevel::Safe    => COLOR_SAFE,
        RiskLevel::Caution => COLOR_CAUTION,
        RiskLevel::High    => COLOR_HIGH,
    }
}

// Completed passes are drawn green and failed ones red, so the pitch reads as
// the player's real outcome map.
fn pass_color(pass: &PlayerPass) -> Color32 {
    if pass.completed {
        COLOR_SAFE
    } else {
        COLOR_HIGH
    }
}

// Returns the pass destination clamped to the pitch boundary.
fn compute_pass_end(
    start_x: f32, start_y: f32, distance_m: f32, direction_deg: f32,
) -> (f32, f32) {
    let angle_rad: f32 = direction_deg.to_radians();
    let end_x: f32 = (start_x + distance_m * angle_rad.cos()).clamp(0.0, PITCH_LENGTH_M);
    let end_y: f32 = (start_y + distance_m * angle_rad.sin()).clamp(0.0, PITCH_WIDTH_M);
    (end_x, end_y)
}

// Runs the full inference pipeline from current slider state.
// Returns None when the model failed to load.
fn compute_inference(
    model_state: &ModelState,
    sliders:     &SliderState,
) -> Option<InferenceResult> {
    let model: &PassModel = match model_state {
        ModelState::Loaded(m) => m,
        ModelState::Failed(_) => return None,
    };

    let (end_x, end_y): (f32, f32) = compute_pass_end(
        sliders.start_x, sliders.start_y, sliders.distance_m, sliders.direction_deg,
    );

    let angle_deg: f32 =
        angle_to_goal_degrees(sliders.start_x, sliders.start_y, end_x, end_y);

    // Map 0–5 defenders linearly to 0.0–1.0 pressure (training feature range).
    let pressure: f32 = (sliders.defenders / FULL_PRESSURE_DEFENDERS).clamp(0.0, 1.0);

    let input: PassInput = PassInput {
        distance_m: sliders.distance_m as f64,
        angle_deg:  angle_deg as f64,
        pressure:   pressure as f64,
    };

    let xp: f32         = model.predict(&input) as f32;
    let risk: RiskLevel = classify_risk(xp);
    let color: Color32  = risk_to_color(&risk);

    Some(InferenceResult { xp, risk, color })
}

// ---- Pitch transform --------------------------------------------------------

impl PitchTransform {
    fn from_rect(rect: Rect) -> PitchTransform {
        let scale_x: f32 = rect.width()  / PITCH_LENGTH_M;
        let scale_y: f32 = rect.height() / PITCH_WIDTH_M;
        let scale: f32   = scale_x.min(scale_y);

        let offset_x: f32 = rect.min.x + (rect.width()  - PITCH_LENGTH_M * scale) / 2.0;
        let offset_y: f32 = rect.min.y + (rect.height() - PITCH_WIDTH_M  * scale) / 2.0;

        PitchTransform { offset_x, offset_y, scale }
    }

    fn to_screen(&self, pitch_x: f32, pitch_y: f32) -> Pos2 {
        Pos2 {
            x: self.offset_x + pitch_x * self.scale,
            y: self.offset_y + pitch_y * self.scale,
        }
    }
}

// ---- Player pass helpers ----------------------------------------------------

// Sliders that reproduce exactly the features this pass was scored with:
// same origin, same length, same direction, same binary pressure.
fn sliders_from_pass(pass: &PlayerPass) -> SliderState {
    let delta: Vec2 = Vec2::new(pass.end_x - pass.start_x, pass.end_y - pass.start_y);

    SliderState {
        start_x:       pass.start_x,
        start_y:       pass.start_y,
        distance_m:    delta.length(),
        direction_deg: delta.y.atan2(delta.x).to_degrees(),
        defenders:     if pass.under_pressure { FULL_PRESSURE_DEFENDERS } else { 0.0 },
    }
}

// The ranked player the user picked in the right panel, if any.
fn selected_player<'a>(
    selection: &Selection,
    match_load_state: &'a MatchLoadState,
) -> Option<&'a PlayerStat> {
    let report: &MatchReport = match match_load_state {
        MatchLoadState::Loaded(report) => report,
        MatchLoadState::Empty => return None,
        MatchLoadState::Error(_) => return None,
    };

    selection.player_index.and_then(|index| report.rankings.get(index))
}

// Every pass of the selected player, ready to draw. None when no player is
// selected or no match is loaded.
fn selected_player_passes<'a>(
    selection: &Selection,
    match_load_state: &'a MatchLoadState,
) -> Option<&'a [PlayerPass]> {
    selected_player(selection, match_load_state).map(|stat| stat.passes.as_slice())
}

// Shortest distance from `point` to the segment start→end, in pixels.
fn distance_to_segment(point: Pos2, start: Pos2, end: Pos2) -> f32 {
    let segment: Vec2 = end - start;
    let length_squared: f32 = segment.length_sq();

    if length_squared <= 0.0001 {
        return (point - start).length();
    }

    let projection: f32 = ((point - start).dot(segment) / length_squared).clamp(0.0, 1.0);

    (point - (start + segment * projection)).length()
}

// Index of the pass drawn closest to `point`, when it is within `threshold`
// pixels of the arrow. None when the click misses every arrow.
fn pick_pass(
    passes: &[PlayerPass],
    t: &PitchTransform,
    point: Pos2,
    threshold: f32,
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;

    for (index, pass) in passes.iter().enumerate() {
        let start: Pos2 = t.to_screen(pass.start_x, pass.start_y);
        let end: Pos2 = t.to_screen(pass.end_x, pass.end_y);
        let distance: f32 = distance_to_segment(point, start, end);

        let is_closer: bool = match best {
            Some((_index, best_distance)) => distance < best_distance,
            None => true,
        };
        if distance <= threshold && is_closer {
            best = Some((index, distance));
        }
    }

    best.map(|(index, _distance)| index)
}

// ---- Drawing leaf functions -------------------------------------------------

fn draw_pitch(painter: &egui::Painter, t: &PitchTransform) {
    let grass: Color32  = Color32::from_rgb(34, 100, 44);
    let line:  Color32  = Color32::WHITE;
    let lw:    f32      = 1.5;
    let sk:    StrokeKind = StrokeKind::Middle;

    painter.rect_filled(
        Rect::from_min_max(t.to_screen(0.0, 0.0), t.to_screen(120.0, 80.0)),
        0.0,
        grass,
    );
    painter.rect_stroke(
        Rect::from_min_max(t.to_screen(0.0, 0.0), t.to_screen(120.0, 80.0)),
        0.0, Stroke::new(lw, line), sk,
    );
    painter.line_segment(
        [t.to_screen(60.0, 0.0), t.to_screen(60.0, 80.0)],
        Stroke::new(lw, line),
    );
    painter.circle_stroke(t.to_screen(60.0, 40.0), 9.15 * t.scale, Stroke::new(lw, line));
    painter.circle_filled(t.to_screen(60.0, 40.0), 2.5, line);

    // Penalty areas
    painter.rect_stroke(
        Rect::from_min_max(t.to_screen(0.0, 18.32), t.to_screen(16.5, 61.68)),
        0.0, Stroke::new(lw, line), sk,
    );
    painter.rect_stroke(
        Rect::from_min_max(t.to_screen(103.5, 18.32), t.to_screen(120.0, 61.68)),
        0.0, Stroke::new(lw, line), sk,
    );

    // 6-yard boxes
    painter.rect_stroke(
        Rect::from_min_max(t.to_screen(0.0, 30.34), t.to_screen(5.5, 49.66)),
        0.0, Stroke::new(lw, line), sk,
    );
    painter.rect_stroke(
        Rect::from_min_max(t.to_screen(114.5, 30.34), t.to_screen(120.0, 49.66)),
        0.0, Stroke::new(lw, line), sk,
    );

    // Goals
    painter.rect_stroke(
        Rect::from_min_max(t.to_screen(-2.44, 36.34), t.to_screen(0.0, 43.66)),
        0.0, Stroke::new(lw, line), sk,
    );
    painter.rect_stroke(
        Rect::from_min_max(t.to_screen(120.0, 36.34), t.to_screen(122.44, 43.66)),
        0.0, Stroke::new(lw, line), sk,
    );

    // Penalty spots
    painter.circle_filled(t.to_screen(11.0,  40.0), 2.5, line);
    painter.circle_filled(t.to_screen(109.0, 40.0), 2.5, line);
}

fn draw_arrow_head(
    painter:   &egui::Painter,
    from:      Pos2,
    to:        Pos2,
    color:     Color32,
    width:     f32,
    head_len:  f32,
    head_half: f32,
) {
    let delta: Vec2 = to - from;
    if delta.length() < 1.0 {
        return;
    }
    let dir: Vec2  = delta.normalized();
    let perp: Vec2 = Vec2::new(-dir.y, dir.x); // 90° rotation, no trig

    let base:  Pos2 = to - dir * head_len;
    let left:  Pos2 = base + perp * head_half;
    let right: Pos2 = base - perp * head_half;

    painter.line_segment([to, left],  Stroke::new(width, color));
    painter.line_segment([to, right], Stroke::new(width, color));
}

// One shaft plus its head; sizes are explicit so player passes can be drawn
// smaller than the live simulation arrow.
fn draw_pass_line(
    painter:   &egui::Painter,
    start:     Pos2,
    end:       Pos2,
    color:     Color32,
    width:     f32,
    head_len:  f32,
    head_half: f32,
) {
    painter.line_segment([start, end], Stroke::new(width, color));
    draw_arrow_head(painter, start, end, color, width, head_len, head_half);
}

fn draw_pass_arrow(
    painter: &egui::Painter,
    sliders: &SliderState,
    color:   Color32,
    t:       &PitchTransform,
) {
    let (end_x, end_y): (f32, f32) = compute_pass_end(
        sliders.start_x, sliders.start_y, sliders.distance_m, sliders.direction_deg,
    );

    let screen_start: Pos2 = t.to_screen(sliders.start_x, sliders.start_y);
    let screen_end:   Pos2 = t.to_screen(end_x, end_y);
    let shaft_width:  f32  = 3.0;

    painter.circle_filled(screen_start, 5.5, color);
    draw_pass_line(painter, screen_start, screen_end, color, shaft_width, 14.0, 7.0);
}

// Draws every pass of one player: green when completed, red when failed.
// The pass loaded into the sliders gets a white halo and a thicker head.
fn draw_player_passes(
    painter:  &egui::Painter,
    passes:   &[PlayerPass],
    selected: Option<usize>,
    t:        &PitchTransform,
) {
    let width:     f32 = 2.0;
    let head_len:  f32 = 8.0;
    let head_half: f32 = 4.0;

    for pass in passes {
        let start: Pos2 = t.to_screen(pass.start_x, pass.start_y);
        let end:   Pos2 = t.to_screen(pass.end_x, pass.end_y);
        let color: Color32 = pass_color(pass);

        draw_pass_line(painter, start, end, color, width, head_len, head_half);
        painter.circle_filled(start, 2.0, color);
    }

    let highlighted: Option<&PlayerPass> =
        selected.and_then(|index| passes.get(index));

    if let Some(pass) = highlighted {
        let start: Pos2 = t.to_screen(pass.start_x, pass.start_y);
        let end:   Pos2 = t.to_screen(pass.end_x, pass.end_y);

        draw_pass_line(painter, start, end, Color32::WHITE, 7.0, 13.0, 7.0);
        draw_pass_line(painter, start, end, pass_color(pass), 3.0, 10.0, 5.0);
        painter.circle_filled(start, 4.0, Color32::WHITE);
    }
}

// ---- Panel render functions -------------------------------------------------

fn show_top_panel(ctx: &egui::Context, result: &Option<InferenceResult>) {
    egui::TopBottomPanel::top("xp_status")
        .min_height(48.0)
        .show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_space(16.0);
                ui.label(RichText::new("PassIQ").font(FontId::proportional(22.0)).strong());
                ui.add_space(20.0);
                ui.separator();
                ui.add_space(20.0);

                match result {
                    None => {
                        ui.colored_label(Color32::RED, "⚠  Model not loaded.");
                    }
                    Some(r) => {
                        let xp_text: String  = format!("xP  {:.1}%", r.xp * 100.0);
                        let risk_label: &str = match &r.risk {
                            RiskLevel::Safe    => "Safe pass",
                            RiskLevel::Caution => "Moderate risk",
                            RiskLevel::High    => "High risk — line-breaker",
                        };
                        ui.label(
                            RichText::new(&xp_text)
                                .font(FontId::proportional(26.0))
                                .color(r.color)
                                .strong(),
                        );
                        ui.add_space(24.0);
                        ui.label(
                            RichText::new(risk_label)
                                .font(FontId::proportional(16.0))
                                .color(r.color),
                        );
                    }
                }
            });
            ui.add_space(6.0);
        });
}

fn show_left_panel(
    ctx:              &egui::Context,
    sliders:          &mut SliderState,
    model_state:      &mut ModelState,
    match_path_input: &mut String,
    match_load_state: &mut MatchLoadState,
    selection:        &mut Selection,
) {
    egui::SidePanel::left("controls")
        .min_width(240.0)
        .max_width(280.0)
        .show(ctx, |ui| {
            ui.add_space(12.0);

            // ---- Pass origin -----------------------------------------------
            ui.label(RichText::new("Pass origin").strong());
            ui.add_space(4.0);

            // Sliders only clamp and round the values they edit. A pass loaded
            // from the pitch keeps its exact geometry until the user moves a
            // handle, so the top bar keeps scoring the same pass.
            let clamping: egui::SliderClamping = egui::SliderClamping::Edits;

            let mut sliders_edited: bool =
                ui.add(egui::Slider::new(&mut sliders.start_x, 0.0..=120.0).text("x (m)").step_by(0.5).clamping(clamping)).changed();
            sliders_edited |=
                ui.add(egui::Slider::new(&mut sliders.start_y, 0.0..=80.0).text("y (m)").step_by(0.5).clamping(clamping)).changed();

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(10.0);

            // ---- Pass parameters -------------------------------------------
            ui.label(RichText::new("Pass parameters").strong());
            ui.add_space(4.0);
            sliders_edited |=
                ui.add(egui::Slider::new(&mut sliders.distance_m, 0.0..=MAX_PASS_DISTANCE_M).text("Distance (m)").step_by(0.5).clamping(clamping)).changed();
            sliders_edited |=
                ui.add(egui::Slider::new(&mut sliders.direction_deg, -180.0..=180.0).text("Direction (°)").step_by(1.0).clamping(clamping)).changed();
            sliders_edited |=
                ui.add(egui::Slider::new(&mut sliders.defenders, 0.0..=FULL_PRESSURE_DEFENDERS).text("Defenders").step_by(0.5).clamping(clamping)).changed();

            // Moving a slider detaches the highlighted pass: from then on the
            // pitch models the sliders, not the clicked pass. Setting the
            // sliders programmatically is not an edit.
            if sliders_edited {
                selection.pass_index = None;
            }

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(10.0);

            // ---- Model reload ----------------------------------------------
            ui.label(RichText::new("Model").strong());
            ui.add_space(4.0);
            if ui.button("⟳  Reload weights").clicked() {
                *model_state = load_model_state();
            }
            ui.add_space(4.0);
            match model_state {
                ModelState::Failed(msg) => {
                    // Long absolute paths are wrapped instead of clipped by the panel.
                    ui.add(
                        egui::Label::new(RichText::new(msg.as_str()).color(Color32::RED)).wrap(),
                    );
                }
                ModelState::Loaded(_) => {
                    ui.colored_label(COLOR_SAFE, "Model loaded");
                }
            }

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(10.0);

            // ---- Load match data -------------------------------------------
            ui.label(RichText::new("Load match data").strong());
            ui.add_space(4.0);
            ui.label("StatsBomb events JSON path:");
            ui.add_space(2.0);
            ui.text_edit_singleline(match_path_input);
            ui.add_space(6.0);

            // Quick-load buttons for the pre-downloaded real matches
            ui.label(RichText::new("Quick load:").color(Color32::GRAY).small());
            ui.add_space(2.0);
            for (label, path) in QUICK_MATCHES {
                if ui.small_button(*label).clicked() {
                    *match_path_input = path.to_string();
                }
            }
            ui.add_space(6.0);

            let can_load: bool = matches!(model_state, ModelState::Loaded(_));

            ui.add_enabled_ui(can_load, |ui| {
                if ui.button("▶  Analyze match").clicked() {
                    if let ModelState::Loaded(model) = model_state {
                        // A new report invalidates both selection indices.
                        *selection = Selection::default();
                        *match_load_state = match load_and_analyze_match(match_path_input, model) {
                            Ok(report) => MatchLoadState::Loaded(report),
                            Err(msg)   => MatchLoadState::Error(msg),
                        };
                    }
                }
            });

            if !can_load {
                ui.add_space(2.0);
                ui.colored_label(Color32::GRAY, "Load model weights first.");
            }

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(10.0);

            // ---- Legend ----------------------------------------------------
            ui.label(RichText::new("Legend").strong());
            ui.add_space(4.0);
            ui.colored_label(COLOR_SAFE, "■  Safe      xP > 70%");
            ui.colored_label(COLOR_CAUTION, "■  Moderate  35–70%");
            ui.colored_label(COLOR_HIGH, "■  High risk xP < 35%");
        });
}

// Renders the per-player ranking table. Clicking a name selects that player,
// so the central pitch can draw every pass of that player.
fn show_right_panel(
    ctx: &egui::Context,
    match_load_state: &MatchLoadState,
    selection: &mut Selection,
) {
    egui::SidePanel::right("match_report")
        .min_width(320.0)
        .max_width(400.0)
        .show(ctx, |ui| {
            ui.add_space(12.0);
            ui.label(RichText::new("Match Analysis").font(FontId::proportional(18.0)).strong());
            ui.add_space(8.0);

            match match_load_state {
                MatchLoadState::Empty => {
                    ui.add_space(16.0);
                    ui.colored_label(
                        Color32::GRAY,
                        "No match loaded.\nEnter the path to a StatsBomb\nevents JSON file and click\n▶ Analyze match.",
                    );
                }

                MatchLoadState::Error(msg) => {
                    ui.add_space(8.0);
                    ui.colored_label(Color32::RED, format!("⚠  {msg}"));
                }

                MatchLoadState::Loaded(report) => {
                    show_report_content(ui, report, selection);
                }
            }
        });
}

// Renders the table content once a report is loaded.
// Separated so show_right_panel stays readable.
fn show_report_content(ui: &mut egui::Ui, report: &MatchReport, selection: &mut Selection) {
    // ---- Summary header ----------------------------------------------------
    let file_name: &str = report
        .match_path
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(&report.match_path);

    ui.label(
        RichText::new(file_name)
            .font(FontId::proportional(12.0))
            .color(Color32::GRAY),
    );
    ui.add_space(4.0);
    ui.label(format!(
        "{} passes  ·  {} players",
        report.total_passes, report.player_count
    ));

    let chosen: Option<&PlayerStat> =
        selection.player_index.and_then(|index| report.rankings.get(index));
    if let Some(stat) = chosen {
        ui.add_space(6.0);
        ui.label(
            RichText::new(format!(
                "{} — {} passes on the pitch.\nClick an arrow to load it into the sliders.",
                stat.player,
                stat.passes.len()
            ))
            .small()
            .color(Color32::GRAY),
        );
        ui.horizontal(|ui| {
            ui.colored_label(COLOR_SAFE, "■  Completed");
            ui.colored_label(COLOR_HIGH, "■  Failed");
        });

        let loaded: Option<&PlayerPass> =
            selection.pass_index.and_then(|index| stat.passes.get(index));
        if let Some(pass) = loaded {
            let outcome: &str = if pass.completed { "completed" } else { "failed" };
            ui.label(
                RichText::new(format!(
                    "Loaded pass: xP {:.1}% · {outcome}",
                    pass.xp * 100.0
                ))
                .small()
                .color(pass_color(pass)),
            );
        }
    }

    ui.add_space(10.0);
    ui.separator();
    ui.add_space(8.0);

    // ---- Rankings table ----------------------------------------------------
    egui::ScrollArea::vertical()
        .id_salt("rankings_scroll")
        .show(ui, |ui| {
            egui::Grid::new("player_rankings")
                .num_columns(5)
                .spacing([10.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    // Header row
                    ui.label(RichText::new("#").strong());
                    ui.label(RichText::new("Player").strong());
                    ui.label(RichText::new("Passes").strong());
                    ui.label(RichText::new("Avg xP").strong());
                    ui.label(RichText::new("Comp.").strong());
                    ui.end_row();

                    for (index, stat) in report.rankings.iter().enumerate() {
                        show_player_row(ui, stat, index, selection);
                    }
                });
        });
}

// Renders one row of the ranking table. The name is a toggle: clicking it
// selects the player's passes; clicking it again clears the selection.
fn show_player_row(
    ui: &mut egui::Ui,
    stat: &PlayerStat,
    index: usize,
    selection: &mut Selection,
) {
    let xp_color: Color32 = risk_to_color(&classify_risk(stat.avg_xp));

    // Truncate long player names at 20 Unicode characters
    let display_name: String = stat.player.chars().take(20).collect();
    let is_selected: bool = selection.player_index == Some(index);

    ui.label(stat.rank.to_string());
    if ui.selectable_label(is_selected, &display_name).clicked() {
        *selection = Selection::default();
        if !is_selected {
            selection.player_index = Some(index);
        }
    }
    ui.label(stat.total_passes.to_string());
    ui.label(
        RichText::new(format!("{:.1}%", stat.avg_xp * 100.0)).color(xp_color),
    );
    ui.label(format!("{}/{}", stat.completed, stat.total_passes));
    ui.end_row();
}

fn show_central_panel(
    ctx:              &egui::Context,
    sliders:          &mut SliderState,
    result:           &Option<InferenceResult>,
    match_load_state: &MatchLoadState,
    selection:        &mut Selection,
) {
    egui::CentralPanel::default().show(ctx, |ui| {
        let available_rect: Rect      = ui.available_rect_before_wrap();
        let transform: PitchTransform = PitchTransform::from_rect(available_rect);
        let painter: &egui::Painter   = ui.painter();

        draw_pitch(painter, &transform);

        let arrow_color: Color32 = match result {
            Some(r) => r.color,
            None    => Color32::GRAY,
        };

        let player_passes: Option<&[PlayerPass]> =
            selected_player_passes(selection, match_load_state);

        match player_passes {
            Some(passes) => {
                draw_player_passes(painter, passes, selection.pass_index, &transform);

                // With a pass highlighted, that pass is what the sliders show.
                if selection.pass_index.is_none() {
                    draw_pass_arrow(painter, sliders, arrow_color, &transform);
                }

                handle_pass_click(
                    ui, available_rect, passes, &transform, sliders, selection,
                );
            }
            None => draw_pass_arrow(painter, sliders, arrow_color, &transform),
        }
    });
}

// Turns a click on the pitch into "load this pass into the sliders".
fn handle_pass_click(
    ui:          &egui::Ui,
    pitch_rect:  Rect,
    passes:      &[PlayerPass],
    transform:   &PitchTransform,
    sliders:     &mut SliderState,
    selection:   &mut Selection,
) {
    let response: egui::Response = ui.interact(
        pitch_rect,
        ui.id().with("player_pass_click"),
        Sense::click(),
    );

    if !response.clicked() {
        return;
    }

    let pointer: Pos2 = match response.interact_pointer_pos() {
        Some(position) => position,
        None => return,
    };

    if let Some(index) = pick_pass(passes, transform, pointer, PASS_PICK_RADIUS_PX) {
        *sliders = sliders_from_pass(&passes[index]);
        selection.pass_index = Some(index);
    }
}

// ---- Model loading ----------------------------------------------------------

pub fn load_model_state() -> ModelState {
    let candidates: Vec<PathBuf> = paths::weights_candidates();

    for path in &candidates {
        match PassModel::load(path.as_path()) {
            Ok(model) => return ModelState::Loaded(model),
            // A missing file only means "try the next candidate". Any other
            // failure must surface: a malformed weights file is not skipped.
            Err(LoadError::FileNotFound { .. }) => continue,
            Err(error) => return ModelState::Failed(error.to_string()),
        }
    }

    let tried: String = candidates
        .iter()
        .map(|path| format!("  {}", path.display()))
        .collect::<Vec<String>>()
        .join("\n");

    ModelState::Failed(format!("model_weights.json not found.\nSearched:\n{tried}"))
}

// ---- App impl ---------------------------------------------------------------

impl PassIQApp {
    pub fn new() -> PassIQApp {
        PassIQApp {
            model_state:      load_model_state(),
            sliders: SliderState {
                start_x:       50.0,
                start_y:       40.0,
                distance_m:    20.0,
                direction_deg:  0.0,
                defenders:      1.0,
            },
            // Default path: El Clásico Real Madrid 2-1 Barcelona (La Liga 2020/21)
            match_path_input: "data/real/3773497.json".to_owned(),
            match_load_state: MatchLoadState::Empty,
            selection: Selection::default(),
        }
    }
}

impl eframe::App for PassIQApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Stage 1: pure inference — produces owned value, no borrow of self remains.
        let result: Option<InferenceResult> =
            compute_inference(&self.model_state, &self.sliders);

        // Stages 2–5: draw panels in order, passing data explicitly.
        // right panel MUST come before central panel (egui layout constraint).
        show_top_panel(ctx, &result);
        show_left_panel(
            ctx,
            &mut self.sliders,
            &mut self.model_state,
            &mut self.match_path_input,
            &mut self.match_load_state,
            &mut self.selection,
        );
        show_right_panel(ctx, &self.match_load_state, &mut self.selection);
        show_central_panel(
            ctx,
            &mut self.sliders,
            &result,
            &self.match_load_state,
            &mut self.selection,
        );

        ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    // Regression: model_weights.json used to be resolved only relative to the
    // process directory, so opening target\release\passiq.exe (double-click or
    // shortcut) left the model unloaded. From the release directory the weights
    // must now be found at the project root.
    #[test]
    fn load_model_state_finds_weights_from_release_dir() {
        let project_dir: PathBuf = env::current_dir().expect("read process directory");
        let release_dir: PathBuf = project_dir.join("target").join("release");
        fs::create_dir_all(&release_dir).expect("create target/release");

        env::set_current_dir(&release_dir).expect("enter target/release");
        let state: ModelState = load_model_state();
        env::set_current_dir(&project_dir).expect("restore process directory");

        assert!(
            matches!(state, ModelState::Loaded(_)),
            "model did not load from target/release"
        );
    }

    // The default match path is project-root relative, so it must resolve and
    // analyze even when the process directory is somewhere else.
    #[test]
    fn default_match_path_resolves_and_analyzes() {
        let state: ModelState = load_model_state();
        let model: &PassModel = match &state {
            ModelState::Loaded(model) => model,
            ModelState::Failed(message) => panic!("model not loaded: {message}"),
        };

        let report: MatchReport =
            load_and_analyze_match("data/real/3773497.json", model).expect("analyze default match");

        assert!(report.total_passes > 0, "no passes analyzed");
        assert!(report.player_count > 0, "no players ranked");
    }

    fn pass_from(start_x: f32, start_y: f32, end_x: f32, end_y: f32) -> PlayerPass {
        PlayerPass {
            start_x,
            start_y,
            end_x,
            end_y,
            under_pressure: false,
            completed: true,
            xp: 0.5,
        }
    }

    // Clicking a pass must fill the sliders with that pass's exact features.
    #[test]
    fn sliders_from_pass_rebuild_the_pass_geometry() {
        let mut pass: PlayerPass = pass_from(30.0, 20.0, 65.0, 45.0);
        pass.under_pressure = true;

        let sliders: SliderState = sliders_from_pass(&pass);

        assert!((sliders.start_x - 30.0).abs() < 0.001);
        assert!((sliders.start_y - 20.0).abs() < 0.001);
        assert!((sliders.distance_m - 43.0116).abs() < 0.01);
        assert!((sliders.direction_deg - 35.5377).abs() < 0.01);
        assert!((sliders.defenders - FULL_PRESSURE_DEFENDERS).abs() < 0.001);
    }

    // Clicking near an arrow returns that pass; clicking away returns nothing.
    #[test]
    fn pick_pass_selects_the_arrow_under_the_pointer() {
        // 600x400 px for a 120x80 m pitch → scale 5 px/m, no centring offset.
        let pitch: Rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(600.0, 400.0));
        let transform: PitchTransform = PitchTransform::from_rect(pitch);
        let passes: Vec<PlayerPass> = vec![
            pass_from(10.0, 10.0, 20.0, 10.0),
            pass_from(60.0, 40.0, 80.0, 40.0),
        ];

        let on_second: Option<usize> =
            pick_pass(&passes, &transform, Pos2::new(350.0, 203.0), PASS_PICK_RADIUS_PX);
        let empty_grass: Option<usize> =
            pick_pass(&passes, &transform, Pos2::new(350.0, 260.0), PASS_PICK_RADIUS_PX);

        assert_eq!(on_second, Some(1));
        assert_eq!(empty_grass, None);
    }

    // Loading a real pass into the sliders must score it exactly as the batch
    // analysis did, so the top bar and the highlighted arrow agree.
    #[test]
    fn selected_pass_reproduces_its_model_score() {
        let state: ModelState = load_model_state();
        let model: &PassModel = match &state {
            ModelState::Loaded(model) => model,
            ModelState::Failed(message) => panic!("model not loaded: {message}"),
        };
        let report: MatchReport =
            load_and_analyze_match("data/real/3773497.json", model).expect("analyze default match");
        let stat: &PlayerStat = report
            .rankings
            .iter()
            .max_by_key(|player| player.passes.len())
            .expect("a ranked player");

        for pass in &stat.passes {
            let sliders: SliderState = sliders_from_pass(pass);
            let result: InferenceResult =
                compute_inference(&state, &sliders).expect("inference");

            assert!(
                (result.xp - pass.xp).abs() < 0.0005,
                "sliders scored {:.5} but the pass was {:.5}",
                result.xp,
                pass.xp
            );
        }
    }
}
