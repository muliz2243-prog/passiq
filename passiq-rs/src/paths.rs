// ============================================================
// PassIQ — paths.rs
// Working-directory independent path resolution.
//
// The executable is normally opened by double-click, so the process
// directory is target\release\ (or any other folder) instead of the
// project root. Every relative path the app reads goes through this
// module: the project root is the nearest ancestor of the executable
// or of the process directory that contains model_weights.json.
// ============================================================

use std::env;
use std::path::{Component, Path, PathBuf};

const ROOT_MARKER: &str = "model_weights.json";
const MAX_ANCESTOR_DEPTH: usize = 6;

// Returns the project root, or None when the marker file is above neither
// the executable nor the process directory.
pub fn project_root() -> Option<PathBuf> {
    let from_exe: Option<PathBuf> = executable_dir().and_then(|dir| find_root_from(dir.as_path()));
    let from_cwd: Option<PathBuf> =
        env::current_dir().ok().and_then(|dir| find_root_from(dir.as_path()));

    from_exe.or(from_cwd)
}

// Pure: walks up from `start` looking for ROOT_MARKER. The depth is capped so
// a missing project never turns into a full-disk scan.
fn find_root_from(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .take(MAX_ANCESTOR_DEPTH)
        .find(|dir| dir.join(ROOT_MARKER).is_file())
        .map(Path::to_path_buf)
}

fn executable_dir() -> Option<PathBuf> {
    let exe_path: PathBuf = env::current_exe().ok()?;

    exe_path.parent().map(Path::to_path_buf)
}

// Candidate locations for model_weights.json, in priority order:
//   1. the process directory (cargo run from passiq-rs\)
//   2. the executable directory (weights shipped next to passiq.exe)
//   3. the project root (double-click, shortcut, any other directory)
pub fn weights_candidates() -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(process_dir) = env::current_dir() {
        candidates.push(process_dir.join(ROOT_MARKER));
    }
    if let Some(exe_dir) = executable_dir() {
        candidates.push(exe_dir.join(ROOT_MARKER));
    }
    if let Some(root) = project_root() {
        candidates.push(root.join(ROOT_MARKER));
    }

    deduplicate(candidates)
}

// Pure: drops repeated paths while keeping the original priority order.
fn deduplicate(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut unique: Vec<PathBuf> = Vec::with_capacity(paths.len());

    for path in paths {
        if !unique.contains(&path) {
            unique.push(path);
        }
    }

    unique
}

// Pure: joins `input` onto `root`, dropping the leading ".." segments that
// are written for a process started inside passiq-rs\.
pub fn join_from_root(root: &Path, input: &str) -> PathBuf {
    let trimmed: PathBuf = Path::new(input)
        .components()
        .skip_while(|component| matches!(component, Component::ParentDir))
        .collect();

    root.join(trimmed)
}

// Driver: returns the first existing file, trying the path as written
// (relative to the process directory) and then relative to the project root.
pub fn resolve_existing(input: &str) -> Option<PathBuf> {
    resolve_with_root(project_root().as_deref(), input)
}

// Pure: resolution rule used by resolve_existing. Split out so the rule can
// be checked without touching the process directory.
fn resolve_with_root(root: Option<&Path>, input: &str) -> Option<PathBuf> {
    let as_written: PathBuf = PathBuf::from(input);
    let resolved: Option<PathBuf> = if as_written.is_file() {
        Some(as_written)
    } else if as_written.is_absolute() {
        None
    } else {
        root.map(|root_path| join_from_root(root_path, input))
            .filter(|candidate| candidate.is_file())
    };

    resolved
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU32, Ordering};

    static FIXTURE_COUNTER: AtomicU32 = AtomicU32::new(0);

    // Builds <temp>\passiq_paths_<pid>_<n>\project with model_weights.json,
    // data\real\ and passiq-rs\target\release\, mirroring the real layout.
    // Returns (project_root, release_dir); the caller removes project_root.
    fn fixture_tree() -> (PathBuf, PathBuf) {
        let nonce: u32 = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root: PathBuf = env::temp_dir()
            .join(format!("passiq_paths_{}_{}", std::process::id(), nonce))
            .join("project");
        let release_dir: PathBuf = root.join("passiq-rs").join("target").join("release");

        fs::create_dir_all(&release_dir).expect("create release dir");
        fs::create_dir_all(root.join("data").join("real")).expect("create data dir");
        fs::write(root.join(ROOT_MARKER), "{}").expect("write marker file");

        (root, release_dir)
    }

    fn remove_fixture(root: &Path) {
        if let Some(parent) = root.parent() {
            let _ = fs::remove_dir_all(parent);
        }
    }

    #[test]
    fn find_root_from_release_dir_walks_up_to_project_root() {
        let (root, release_dir) = fixture_tree();

        let found: Option<PathBuf> = find_root_from(release_dir.as_path());

        assert_eq!(found.as_deref(), Some(root.as_path()));
        remove_fixture(&root);
    }

    #[test]
    fn find_root_from_markerless_tree_returns_none() {
        let (root, release_dir) = fixture_tree();
        fs::remove_file(root.join(ROOT_MARKER)).expect("remove marker file");

        let found: Option<PathBuf> = find_root_from(release_dir.as_path());

        assert_eq!(found, None);
        remove_fixture(&root);
    }

    #[test]
    fn join_from_root_drops_leading_parent_dirs() {
        let (root, _release_dir) = fixture_tree();

        let resolved: PathBuf = join_from_root(root.as_path(), "../data/real/3773497.json");

        assert_eq!(resolved, root.join("data").join("real").join("3773497.json"));
        remove_fixture(&root);
    }

    #[test]
    fn join_from_root_keeps_plain_relative_paths() {
        let (root, _release_dir) = fixture_tree();

        let resolved: PathBuf = join_from_root(root.as_path(), "data/sample_match.json");

        assert_eq!(resolved, root.join("data").join("sample_match.json"));
        remove_fixture(&root);
    }

    #[test]
    fn resolve_with_root_finds_file_below_project_root() {
        let (root, _release_dir) = fixture_tree();
        let data_file: PathBuf = root.join("data").join("real").join("match.json");
        fs::write(&data_file, "[]").expect("write data file");

        let resolved: Option<PathBuf> =
            resolve_with_root(Some(root.as_path()), "data/real/match.json");

        assert_eq!(resolved.as_deref(), Some(data_file.as_path()));
        remove_fixture(&root);
    }

    #[test]
    fn resolve_with_root_reports_missing_file() {
        let (root, _release_dir) = fixture_tree();

        let resolved: Option<PathBuf> =
            resolve_with_root(Some(root.as_path()), "data/real/absent.json");

        assert_eq!(resolved, None);
        remove_fixture(&root);
    }
}
