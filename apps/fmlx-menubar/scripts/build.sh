#!/usr/bin/env bash
# build.sh - produce "Flyto MLX.app" from the Rust menubar.
#
# The app embeds no Python. It runs the host venv
# (~/Code/omlx/.venv/bin/python -m omlx.cli serve), so editing a .py in the
# worktree and restarting the server from the menu applies it.
#
# Usage:
#   apps/fmlx-menubar/scripts/build.sh            # -> apps/fmlx-menubar/build/Flyto MLX.app
#   FMLX_APP_OUT=/some/dir apps/fmlx-menubar/scripts/build.sh
#
# Install (stop the old app first, see README.md):
#   ditto "build/Flyto MLX.app" "/Applications/Flyto MLX.app"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CRATE_DIR/../.." && pwd)"
OUT_DIR="${FMLX_APP_OUT:-$CRATE_DIR/build}"
APP="$OUT_DIR/Flyto MLX.app"

VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$REPO_ROOT/omlx/_version.py")"
VERSION="${VERSION:-0.0.0}"

echo "[build] cargo build --release"
(cd "$CRATE_DIR" && cargo build --release)
BIN="$CRATE_DIR/target/release/fmlx-menubar"

echo "[build] staging $APP (version $VERSION)"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/fmlx-menubar"

ICONSET="$(mktemp -d)/AppIcon.iconset"
"$BIN" --render-app-icon "$ICONSET"
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
rm -rf "$(dirname "$ICONSET")"

# No LSUIElement: the app starts as a regular app and switches itself to an
# accessory (no Dock icon) once the status item exists. LSUIElement plus that
# switch can leave the status item hidden on macOS Tahoe and later.
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.panwudi.flyto-mlx</string>
    <key>CFBundleName</key>
    <string>Flyto MLX</string>
    <key>CFBundleDisplayName</key>
    <string>Flyto MLX</string>
    <key>CFBundleExecutable</key>
    <string>fmlx-menubar</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

codesign --force --sign - "$APP"
xattr -cr "$APP"
echo "[build] done: $APP"
