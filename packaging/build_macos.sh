#!/usr/bin/env bash
# ============================================================
#  Stars2Cells macOS build
#
#  Usage (from anywhere, any Python 3.10-3.12):
#      ./packaging/build_macos.sh [version]
#
#  Produces:
#      dist/Stars2Cells.app
#      Stars2Cells_<version>_macos_<arch>.zip
#      Stars2Cells_<version>_macos_<arch>.dmg
#
#  Optional signing / notarization (set env vars before running):
#      SIGN_IDENTITY="Developer ID Application: Ari Peden-Asarch (2R3GA8BS26)"
#      NOTARY_PROFILE="S2C"        # keychain profile from `xcrun notarytool store-credentials`
#
#  Unsigned builds still work: users right-click the app > Open the
#  first time to get past Gatekeeper.
# ============================================================
set -euo pipefail

VERSION="${1:-1.0.0}"
export S2C_VERSION="$VERSION"

cd "$(dirname "$0")/.."

ARCH="$(uname -m)"          # arm64 or x86_64
RELEASE_DIR="Stars2Cells_${VERSION}_release"
ZIP_NAME="Stars2Cells_${VERSION}_macos_${ARCH}.zip"
DMG_NAME="Stars2Cells_${VERSION}_macos_${ARCH}.dmg"

echo "============================================"
echo "  Stars2Cells ${VERSION} - macOS ${ARCH} build"
echo "============================================"

# 1) Fresh build venv
rm -rf build_env
python3 -m venv build_env
source build_env/bin/activate
python -m pip install --upgrade pip
pip install -r packaging/requirements-build.txt

# 2) Icons from S2C_logo.png
python packaging/make_icons.py

# 3) Build the .app
pyinstaller --noconfirm --clean packaging/stars2cells.spec
test -d "dist/Stars2Cells.app" || { echo "ERROR: dist/Stars2Cells.app not produced"; exit 1; }

# 4) Optional code signing (must happen before dmg/zip)
if [[ -n "${SIGN_IDENTITY:-}" ]]; then
    echo "Signing with: ${SIGN_IDENTITY}"
    /usr/bin/codesign --deep --force --options runtime \
        --sign "${SIGN_IDENTITY}" "dist/Stars2Cells.app"
    /usr/bin/codesign --verify --verbose=2 "dist/Stars2Cells.app"
else
    echo "SIGN_IDENTITY not set - skipping code signing (users must right-click > Open once)"
fi

# 5) Release folder: the .app plus top-level docs
rm -rf "${RELEASE_DIR}"
mkdir -p "${RELEASE_DIR}"
cp -R "dist/Stars2Cells.app" "${RELEASE_DIR}/"
cp README.md LICENSE base_data_requirements.txt exporting_data.txt "${RELEASE_DIR}/"

# Optional auto-restart launcher next to the app
cp "packaging/launchers/Launch Stars2Cells.command" "${RELEASE_DIR}/"
chmod +x "${RELEASE_DIR}/Launch Stars2Cells.command"

# 6) Zip (ditto preserves signatures/resource forks, unlike plain zip -r)
rm -f "${ZIP_NAME}"
/usr/bin/ditto -c -k --keepParent "${RELEASE_DIR}" "${ZIP_NAME}"

# 7) DMG
rm -f "${DMG_NAME}"
hdiutil create -volname "Stars2Cells ${VERSION}" \
    -srcfolder "${RELEASE_DIR}" \
    -ov -format UDZO "${DMG_NAME}"

# 8) Optional notarization + stapling
if [[ -n "${NOTARY_PROFILE:-}" ]]; then
    echo "Submitting ${DMG_NAME} for notarization (profile: ${NOTARY_PROFILE})..."
    xcrun notarytool submit "${DMG_NAME}" --keychain-profile "${NOTARY_PROFILE}" --wait
    xcrun stapler staple "${DMG_NAME}"
else
    echo "NOTARY_PROFILE not set - skipping notarization"
fi

echo
echo "============================================"
echo "  Done."
echo "  App:  dist/Stars2Cells.app"
echo "  Zip:  ${ZIP_NAME}"
echo "  DMG:  ${DMG_NAME}"
echo "============================================"
