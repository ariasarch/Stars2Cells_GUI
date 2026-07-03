# Distributing Stars2Cells (EXE / DMG / ZIP)

This repo ships as fully self-contained builds — end users never install
Python, conda, or any packages. Everything is bundled with
[PyInstaller](https://pyinstaller.org) via `packaging/stars2cells.spec`.

Public downloads are published as GitHub Releases on
**[ariasarch/Stars2Cells](https://github.com/ariasarch/Stars2Cells/releases)**
(builds are produced by this repo's CI or the local scripts, then land there).

Deliverables per release:

| File | Platform | What users do |
|---|---|---|
| `Stars2Cells_<ver>_windows_x64.zip` | Windows 10/11 | Unzip, double-click `Stars2Cells\Stars2Cells.exe` |
| `Stars2Cells_<ver>_macos_arm64.dmg` / `.zip` | Apple Silicon Macs | Open dmg, drag folder anywhere, open `Stars2Cells.app` |
| `Stars2Cells_<ver>_macos_x86_64.dmg` / `.zip` | Intel Macs | Same |

FAISS is bundled from pip wheels; if that ever fails on a machine the app
silently falls back to the scikit-learn KDTree path, same as running from
source.

---

## Option A — Let GitHub build everything (recommended)

You don't need a Windows machine at all, and the Mac side only needs your
laptop for signing/notarizing.

```bash
git tag v1.0.0
git push origin v1.0.0
```

The `Build distributables` workflow (`.github/workflows/build-release.yml`)
builds on Windows, Apple-Silicon macOS, and Intel macOS, then attaches the
zip/dmg files to a GitHub Release. You can also run it manually from the
**Actions** tab (artifacts appear on the run page instead).

Where the release lands depends on one secret:

- **`RELEASE_TOKEN` set** (Settings → Secrets and variables → Actions on
  this repo; a fine-grained PAT with *Contents: read & write* on
  `ariasarch/Stars2Cells`): the release is created/updated on
  **ariasarch/Stars2Cells** — the public download page.
- **No secret**: the release is created on this repo instead; download the
  files and upload them to a Stars2Cells release by hand.

CI builds are **unsigned**. The Windows zip is fine as-is (SmartScreen shows
a one-time "More info → Run anyway"). For the Mac dmg, either tell users to
right-click → Open the first time, or sign + notarize it yourself — see below.

## Option B — Build locally

### Windows exe + zip

On any Windows machine with Python 3.10–3.12 on PATH (no conda needed):

```bat
git clone https://github.com/ariasarch/Stars2Cells_GUI.git
cd Stars2Cells_GUI
packaging\build_windows.bat 1.0.0
```

Output: `dist\Stars2Cells\Stars2Cells.exe` and
`Stars2Cells_1.0.0_windows_x64.zip`.

### macOS app + zip + dmg

```bash
git clone https://github.com/ariasarch/Stars2Cells_GUI.git
cd Stars2Cells_GUI
./packaging/build_macos.sh 1.0.0
```

Output: `dist/Stars2Cells.app`, `Stars2Cells_1.0.0_macos_<arch>.zip`,
`Stars2Cells_1.0.0_macos_<arch>.dmg`.

A Mac only builds for its own architecture — build on an Apple-Silicon Mac
for arm64 and an Intel Mac (or the CI's `macos-13` runner) for x86_64.

---

## Signing + notarizing the Mac build

Same flow as before, folded into the build script. One-time setup for the
notary profile (uses an app-specific password from appleid.apple.com):

```bash
xcrun notarytool store-credentials "S2C" \
  --apple-id "you@example.com" \
  --team-id 2R3GA8BS26
```

Then a fully signed, notarized, stapled build is just:

```bash
SIGN_IDENTITY="Developer ID Application: Ari Peden-Asarch (2R3GA8BS26)" \
NOTARY_PROFILE="S2C" \
./packaging/build_macos.sh 1.0.0
```

The script signs the `.app` **before** creating the zip/dmg, notarizes the
dmg with `notarytool submit --wait`, and staples the ticket.

To sign/notarize a dmg that CI built (download it, extract nothing):
CI zips can't be re-signed cleanly — rebuild locally with the env vars set,
or sign the extracted `.app`, then re-create zip + dmg:

```bash
codesign --deep --force --options runtime \
  --sign "Developer ID Application: Ari Peden-Asarch (2R3GA8BS26)" "Stars2Cells.app"
xcrun notarytool submit Stars2Cells_1.0.0_macos_arm64.dmg --keychain-profile "S2C" --wait
xcrun stapler staple Stars2Cells_1.0.0_macos_arm64.dmg
```

---

## How it fits together

```
packaging/
├── stars2cells.spec        # PyInstaller config (entry: stars2cells.py, onedir)
├── requirements-build.txt  # pip-only deps (numpy/scipy/faiss via pip, no conda)
├── make_icons.py           # S2C_logo.png -> .ico / .icns at build time
├── build_windows.bat       # venv -> deps -> icons -> pyinstaller -> zip
└── build_macos.sh          # venv -> deps -> icons -> pyinstaller -> [sign] -> zip + dmg -> [notarize]
.github/workflows/
└── build-release.yml       # runs the above on tag push / manual dispatch
```

Notes for future maintenance:

- **onedir, not onefile**: the pipeline spawns multiprocessing workers;
  onedir keeps worker startup fast and reliable. Ship the whole folder.
- `stars2cells.py` calls `multiprocessing.freeze_support()` in its
  `__main__` block — do not remove it, or the frozen exe will endlessly
  respawn GUI windows the moment the pipeline starts workers.
- `S2C_logo.png` is bundled into the app root; `utilities/utils.py` already
  searches `__file__`-relative paths so the icon/splash resolve when frozen.
- Bump pins in `packaging/requirements-build.txt` in lockstep with
  `requirements.txt` (they mirror each other, minus conda).
