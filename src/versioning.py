"""Dataset versioning from Python: DVC (data) + Git (metadata + tag).

Used by the Streamlit app (tab "Data versions"):
    version_data(...)   -> dvc add + git add + git commit + git tag   (+ optional push)
    list_versions()     -> every tag "v..." with its message
    load_version(tag)   -> the dataset exactly as it was at that tag
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = "data/dataset.csv"          # the DVC-tracked dataset
VERSIONS_DIR = "reports/versions"       # one small JSON per version (tracked by Git)
TAG_RE = re.compile(r"^v\d+(\.\d+){0,3}$")


# --------------------------------------------------------------------------
# low level
# --------------------------------------------------------------------------
def _run(cmd: list[str], timeout: int = 300) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout: {' '.join(cmd)}"


def project_status() -> dict:
    """Is the project ready for versioning? (git + dvc installed and initialised)"""
    st = {"git_installed": shutil.which("git") is not None, "dvc_installed": shutil.which("dvc") is not None,
          "git_repo": False, "dvc_repo": (ROOT / ".dvc").exists(), "git_user": False,
          "git_remote": False, "dvc_remote": False}
    if st["git_installed"]:
        rc, out = _run(["git", "rev-parse", "--show-toplevel"])
        st["git_repo"] = rc == 0 and Path(out.splitlines()[-1]).resolve() == ROOT.resolve()
        rc, name = _run(["git", "config", "user.name"])
        rc2, mail = _run(["git", "config", "user.email"])
        st["git_user"] = rc == 0 and rc2 == 0 and bool(name) and bool(mail)
        if st["git_repo"]:
            st["git_remote"] = bool(_run(["git", "remote"])[1])
    if st["dvc_installed"] and st["dvc_repo"]:
        st["dvc_remote"] = bool(_run(["dvc", "remote", "list"])[1])
    st["ready"] = all(st[k] for k in ("git_installed", "dvc_installed", "git_repo", "dvc_repo", "git_user"))
    return st


def list_versions() -> list[dict]:
    """Tags starting with v, newest first: [{tag, date, message}]"""
    rc, out = _run(["git", "for-each-ref", "refs/tags", "--sort=-creatordate",
                    "--format=%(refname:short)|%(creatordate:iso8601)|%(contents:subject)"])
    if rc != 0:
        return []
    versions = []
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3 and re.match(r"^v\d", parts[0]):
            versions.append({"tag": parts[0], "date": parts[1], "message": parts[2]})
    return versions


def suggest_next_tag(tags: list[str]) -> str:
    nums = []
    for t in tags:
        m = re.match(r"^v(\d+)\.(\d+)$", t)
        if m:
            nums.append((int(m.group(1)), int(m.group(2))))
    if not nums:
        return "v1.0"
    major, minor = max(nums)
    return f"v{major}.{minor + 1}"


def read_version_info(tag: str) -> dict | None:
    """Parameters / metadata recorded when the version was created (stored in Git)."""
    rc, out = _run(["git", "show", f"{tag}:{VERSIONS_DIR}/{tag}.json"])
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def load_version(tag: str) -> pd.DataFrame:
    """The dataset exactly as it was at `tag` (works without touching the working folder)."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "dataset.csv"
        rc, out = _run(["dvc", "get", ".", DATA_FILE, "--rev", tag, "-o", str(target), "--force"])
        if rc != 0 or not target.exists():
            raise RuntimeError(f"Could not load {tag} from DVC: {out[-400:]}")
        return pd.read_csv(target, low_memory=False)


# --------------------------------------------------------------------------
# create a version
# --------------------------------------------------------------------------
def _md5_of_dvc_file() -> str | None:
    f = ROOT / f"{DATA_FILE}.dvc"
    if not f.exists():
        return None
    m = re.search(r"md5:\s*([0-9a-f]+)", f.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def version_data(df: pd.DataFrame, params: dict, tag: str, message: str, push: bool = False) -> dict:
    """Save `df` as the new tracked dataset and record it: dvc add -> git add -> git commit -> git tag."""
    result: dict = {"success": False, "tag": tag, "steps": [], "error": ""}
    message = (message or "").strip() or f"Dataset version {tag}"

    def fail(msg: str) -> dict:
        result["error"] = msg
        return result

    status = project_status()
    if not status["ready"]:
        missing = [k for k in ("git_installed", "dvc_installed", "git_repo", "dvc_repo", "git_user") if not status[k]]
        return fail("The project is not ready for versioning (missing: " + ", ".join(missing) + ").")
    if not TAG_RE.match(tag or ""):
        return fail("Invalid tag. Use a format like v1.1, v1.2 or v2.0.")
    if tag in [v["tag"] for v in list_versions()]:
        return fail(f"The tag {tag} already exists. Choose a new one.")

    def step(cmd: list[str]) -> bool:
        rc, out = _run(cmd)
        result["steps"].append({"command": " ".join(cmd), "ok": rc == 0, "output": out[-600:]})
        return rc == 0

    path = ROOT / DATA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    result["steps"].append({"command": f"write {DATA_FILE} ({len(df):,} rows x {df.shape[1]} columns)",
                            "ok": True, "output": ""})

    if not step(["dvc", "add", DATA_FILE]):
        return fail("`dvc add` failed (see the log below).")

    md5 = _md5_of_dvc_file()
    info = {"tag": tag, "message": message, "created_at": datetime.now().isoformat(timespec="seconds"),
            "rows": int(len(df)), "columns": int(df.shape[1]), "column_names": list(map(str, df.columns)),
            "dvc_md5": md5, "params": params}
    info_path = ROOT / VERSIONS_DIR / f"{tag}.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    to_add = [f"{DATA_FILE}.dvc", "data/.gitignore", f"{VERSIONS_DIR}/{tag}.json"]
    if not step(["git", "add"] + [p for p in to_add if (ROOT / p).exists()]):
        return fail("`git add` failed.")
    if not step(["git", "commit", "-m", message]):
        return fail("`git commit` failed. If it says 'Author identity unknown', run: "
                    "git config --global user.name \"Your Name\" and git config --global user.email you@mail.com")
    if not step(["git", "tag", "-a", tag, "-m", message]):
        return fail("`git tag` failed.")

    if push:
        if status["dvc_remote"]:
            step(["dvc", "push"])
        else:
            result["steps"].append({"command": "dvc push", "ok": False, "output": "skipped: no DVC remote configured"})
        if status["git_remote"]:
            step(["git", "push"])
            step(["git", "push", "--tags"])
        else:
            result["steps"].append({"command": "git push", "ok": False, "output": "skipped: no Git remote configured"})

    result.update(success=True, md5=md5, rows=int(len(df)), columns=int(df.shape[1]))
    return result


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------
def flatten(d, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}." if isinstance(v, dict) else f"{prefix}{k}"))
    else:
        out[prefix.rstrip(".")] = d
    return out


def params_diff(a: dict | None, b: dict | None) -> pd.DataFrame:
    fa, fb = flatten((a or {}).get("params", {})), flatten((b or {}).get("params", {}))
    rows = [{"parameter": k, "version A": str(fa.get(k, "-")), "version B": str(fb.get(k, "-"))}
            for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    return pd.DataFrame(rows, columns=["parameter", "version A", "version B"])
