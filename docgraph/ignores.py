"""Default ignore patterns + per-ecosystem autodetect.

We layer three tiers of ignore patterns:

1. **UNIVERSAL** — always-on: VCS dirs, OS junk, common lockfiles, env files,
   binary/media extensions. Matches Cursor's built-in default list.
2. **TEMPLATES** — applied only when a marker file (e.g. `package.json`,
   `pom.xml`, `Cargo.toml`) is detected at the root. Each template is a small
   curated subset of the corresponding `github/gitignore` file, focused on
   build-output and dependency directories that an indexer should never read.
3. **User files** — `.gitignore` / `.docgraphignore` / `.cursorindexingignore`
   layered on top by `Config.__post_init__`.

Inline strings rather than vendored files: total <3 KB, simpler loading,
and the patterns rarely change.
"""
from __future__ import annotations

from pathlib import Path

# Always-on patterns. Mirrors Cursor's built-in default list. These directory
# names are unambiguously dependency or cache dirs (no chance they contain user
# source code), so we apply them regardless of the detected ecosystem.
UNIVERSAL: list[str] = [
    # All dotfiles and dotfolders. Covers VCS (.git, .svn, .hg, .bzr), IDE
    # (.idea, .vscode, .vs, .history), OS junk (.DS_Store, .AppleDouble),
    # env files (.env, .env.*), tool caches (.venv, .tox, .pytest_cache,
    # .mypy_cache, .ruff_cache, .gradle, .mvn, .terraform, .dart_tool,
    # .elixir_ls, .metals, .bloop, .stack-work, .ipynb_checkpoints, .conda,
    # .dvc, .Rproj.user, etc.), framework outputs (.next, .nuxt, .svelte-kit,
    # .astro, .vercel, .netlify, .turbo, .parcel-cache, .angular, .ng,
    # .nyc_output, .swiftpm, .cxx, .pub-cache, .flutter-plugins),
    # Claude/agent state (.claude/, .docgraph/), and anything else dot-prefixed.
    # `.gitignore`, `.docgraphignore`, `.cursorignore`, `.cursor/rules/*.mdc`
    # are still picked up — Config and rules loaders read those by explicit
    # path, not via the ignore spec.
    ".*",
    # IDE / editor
    "*.iml",
    # OS junk
    "Thumbs.db",
    "ehthumbs.db",
    "Desktop.ini",
    "$RECYCLE.BIN/",
    # Lockfiles
    "*.lock",
    "*.lockb",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lockb",
    "composer.lock",
    "Gemfile.lock",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    # Logs / temp
    "*.log",
    "*.tmp",
    "*.bak",
    "*.swp",
    "*.swo",
    # Binaries
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.bin",
    "*.o",
    "*.a",
    "*.class",
    "*.jar",
    "*.war",
    "*.ear",
    "*.beam",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    # Archives
    "*.zip",
    "*.tar",
    "*.tar.*",
    "*.tgz",
    "*.gz",
    "*.7z",
    "*.rar",
    # Media
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.bmp",
    "*.tiff",
    "*.ico",
    "*.svg",
    "*.webp",
    "*.mp3",
    "*.mp4",
    "*.wav",
    "*.avi",
    "*.mov",
    "*.webm",
    "*.flv",
    # Fonts
    "*.ttf",
    "*.otf",
    "*.woff",
    "*.woff2",
    "*.eot",
    # Docs / office
    "*.pdf",
    "*.doc",
    "*.docx",
    "*.xls",
    "*.xlsx",
    "*.ppt",
    "*.pptx",
    # Common repo-root documentation files (informational, not source)
    "README*",
    "readme*",
    "CHANGELOG*",
    "changelog*",
    "CHANGES*",
    "CONTRIBUTING*",
    "contributing*",
    "LICENSE*",
    "license*",
    "LICENCE*",
    "NOTICE*",
    "AUTHORS*",
    "CODEOWNERS",
    # Min / map
    "*.min.js",
    "*.min.css",
    "*.map",
    # --- Unambiguous dependency / cache directories (no chance of source) ---
    # Non-dot-prefixed dep/cache dirs only — anything dot-prefixed is already
    # covered by the `.*` rule above.
    # JS/TS
    "node_modules/",
    "bower_components/",
    "jspm_packages/",
    "web_modules/",
    "*.tsbuildinfo",
    # Python
    "__pycache__/",
    "venv/",
    "*.egg-info/",
    "pip-wheel-metadata/",
    "htmlcov/",
    # Mobile / iOS / Android
    "DerivedData/",
    "xcuserdata/",
    # Haskell
    "dist-newstyle/",
    # Zig
    "zig-cache/",
    "zig-out/",
    # Data science / ML experiment tracking
    "mlruns/",
    "wandb/",
    "lightning_logs/",
]


# Ambiguously-named build/output dirs and ecosystem-specific patterns. Only
# applied when the matching marker file is detected at the repo root, since
# `target/` / `build/` / `out/` / `bin/` could otherwise be legit source dirs
# in unrelated projects.
TEMPLATES: dict[str, list[str]] = {
    "node": [
        "dist/",
        "build/",
        "out/",
        "coverage/",
    ],
    "angular": [
        "out-tsc/",
        "e2e/test-output/",
    ],
    "python": [
        "build/",
        "dist/",
        "*.egg",
        ".eggs/",
        ".coverage",
        ".coverage.*",
    ],
    "maven": [
        "target/",
        "pom.xml.tag",
        "pom.xml.releaseBackup",
        "pom.xml.versionsBackup",
        "pom.xml.next",
    ],
    "gradle": [
        "**/build/",
        "out/",
    ],
    "android": [
        "captures/",
        "local.properties",
        "*.apk",
        "*.aab",
        "*.ap_",
        "*.dex",
    ],
    "rust": [
        "target/",
        "**/*.rs.bk",
    ],
    "go": [
        "vendor/",
        "*.test",
        "*.out",
    ],
    "dotnet": [
        "bin/",
        "obj/",
        "[Dd]ebug/",
        "[Rr]elease/",
        "x64/",
        "x86/",
        "[Aa][Rr][Mm]/",
        "[Aa][Rr][Mm]64/",
        "packages/",
        "TestResults/",
        "*.user",
        "*.suo",
        "*.userprefs",
        "*.pidb",
    ],
    "swift": [
        ".build/",
        "Pods/",
        "*.xcuserstate",
    ],
    "ruby": [
        "vendor/bundle/",
        ".bundle/",
        "tmp/",
        "*.gem",
    ],
    "dart": [
        "build/",
    ],
    "elixir": [
        "_build/",
        "deps/",
    ],
    "scala": [
        "target/",
        "project/target/",
        "project/project/",
    ],
    "php": [
        "vendor/",
    ],
    "terraform": [
        "*.tfstate",
        "*.tfstate.*",
        "crash.log",
    ],
    "unity": [
        "Library/",
        "Temp/",
        "Logs/",
        "UserSettings/",
        "MemoryCaptures/",
    ],
}


# Detection signals: list of (ecosystem_key, [marker glob patterns]).
# Markers are checked at the root via Path.glob (so `*.csproj` works for .NET).
# Order matters only for the "stop at first match per ecosystem" loop below;
# multiple ecosystems can match a single repo (a Maven + Java project pulls
# both templates).
_DETECTORS: list[tuple[str, list[str]]] = [
    ("node", ["package.json"]),
    ("angular", ["angular.json"]),
    ("python", ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"]),
    ("java", ["pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"]),
    ("maven", ["pom.xml"]),
    ("gradle", ["build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"]),
    ("android", ["AndroidManifest.xml", "app/build.gradle", "app/build.gradle.kts"]),
    ("rust", ["Cargo.toml"]),
    ("go", ["go.mod"]),
    ("dotnet", ["*.csproj", "*.fsproj", "*.vbproj", "*.sln", "global.json"]),
    ("swift", ["Package.swift", "*.xcodeproj", "*.xcworkspace"]),
    ("ruby", ["Gemfile", "*.gemspec"]),
    ("dart", ["pubspec.yaml"]),
    ("elixir", ["mix.exs"]),
    ("scala", ["build.sbt", "project/build.properties", "*.sbt"]),
    ("php", ["composer.json"]),
    ("terraform", ["*.tf", "main.tf"]),
    ("unity", ["Assets/", "ProjectSettings/"]),
]


def detect_ecosystems(root: Path) -> list[str]:
    """Scan `root` for marker files; return matching ecosystem keys.

    Cheap: globs the root once per detector. Sub-millisecond on a typical repo.
    Markers are checked at the root level only (no recursion) — a repo with a
    nested Node sub-app inside a Python repo will still pick up Node iff the
    sub-app surfaces `package.json` at the top level (rare).
    """
    found: list[str] = []
    for key, patterns in _DETECTORS:
        for pat in patterns:
            # Path.glob handles both `*.csproj` and plain `package.json`
            try:
                if any(root.glob(pat)):
                    found.append(key)
                    break
            except OSError:
                continue
    return found


def assemble_ignores(root: Path) -> tuple[list[str], list[str]]:
    """Return (patterns, detected_ecosystems) for `root`.

    `patterns` = UNIVERSAL + the union of TEMPLATES for each detected
    ecosystem. Caller is responsible for layering user `.gitignore` /
    `.docgraphignore` / `.cursorindexingignore` on top.
    """
    detected = detect_ecosystems(root)
    patterns: list[str] = list(UNIVERSAL)
    seen: set[str] = set(patterns)
    for eco in detected:
        for pat in TEMPLATES.get(eco, []):
            if pat not in seen:
                patterns.append(pat)
                seen.add(pat)
    return patterns, detected
