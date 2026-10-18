#!/bin/sh
# Wrap the PyInstaller bundle (packaging/build.py) in a Debian package.
#
# Usage: packaging/linux/build_deb.sh [BUNDLE_DIR] [OUT_DIR]
#        (defaults: dist/transcribe and dist)
# Output: OUT_DIR/transcribe_<version>_<arch>.deb
#
# The bundle carries its own Python, Qt, CTranslate2 and ONNX Runtime, so the package
# installs it as-is under /usr/lib/transcribe and depends only on the system libraries
# the bundle does not include (graphics, xcb, Wayland). The bundled executables must not
# be stripped: PyInstaller appends the Python archive to them.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUNDLE=${1:-$ROOT/dist/transcribe}
OUT=${2:-$ROOT/dist}
[ -x "$BUNDLE/tx" ] && [ -x "$BUNDLE/Transcribe" ] ||
    { echo "error: no bundle in $BUNDLE - run packaging/build.py first" >&2; exit 1; }

PACKAGE=transcribe
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -n 1)
[ -n "$VERSION" ] || { echo "error: no version in pyproject.toml" >&2; exit 1; }
ARCH=$(dpkg --print-architecture)
MAINTAINER="Marcel Ritter <marcel+it+github@linux-ng.de>"
HOMEPAGE=https://github.com/linuxrrze/audio-transcribe
# The commit date, so the same commit always builds the same package.
DATE=$(git -C "$ROOT" log -1 --format=%aD 2>/dev/null || date -R)
export SOURCE_DATE_EPOCH=$(git -C "$ROOT" log -1 --format=%ct 2>/dev/null || date +%s)

# The newest glibc symbol version any bundled binary needs (the build host's glibc at most).
GLIBC=$(find "$BUNDLE" -type f \( -name '*.so*' -o -perm -u+x \) -exec objdump -T {} + \
            2>/dev/null | sed -n 's/.*GLIBC_\([0-9][0-9.]*\).*/\1/p' | sort -uV | tail -n 1)
[ -n "$GLIBC" ] || { echo "error: could not determine the glibc version" >&2; exit 1; }

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
LIB=$STAGE/usr/lib/$PACKAGE
DOC=$STAGE/usr/share/doc/$PACKAGE
mkdir -p "$LIB" "$STAGE/usr/bin" "$STAGE/usr/share/applications" "$DOC" \
    "$STAGE/usr/share/lintian/overrides" "$STAGE/DEBIAN"
cp -a "$BUNDLE/." "$LIB/"
# PyInstaller finds its files through /proc/self/exe, so symlinks work.
ln -s ../lib/$PACKAGE/tx "$STAGE/usr/bin/tx"
ln -s ../lib/$PACKAGE/Transcribe "$STAGE/usr/bin/tx-gui"

cat > "$STAGE/usr/share/applications/$PACKAGE.desktop" <<END
[Desktop Entry]
Type=Application
Name=Transcribe
GenericName=Audio Transcription
GenericName[de]=Audio-Transkription
Comment=Transcribe speech locally and recognise speakers
Comment[de]=Sprache lokal transkribieren und Sprecher erkennen
Exec=tx-gui
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Audio;Utility;
Keywords=transcription;speech;whisper;speaker;diarization;
END

cat > "$DOC/copyright" <<END
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: transcribe
Source: $HOMEPAGE

Files: *
Copyright: 2026 Marcel Ritter
License: AGPL-3+
Comment: /usr/lib/$PACKAGE is a PyInstaller bundle. It also contains Python and
 third-party libraries (Qt/PySide6, CTranslate2, ONNX Runtime, sherpa-onnx,
 faster-whisper, PyAV/FFmpeg, PortAudio and others) under their own licenses;
 their license files are in /usr/lib/$PACKAGE/_internal.

License: AGPL-3+
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as published
 by the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 .
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU Affero General Public License for more details.
 .
 You should have received a copy of the GNU Affero General Public License
 along with this program. If not, see <https://www.gnu.org/licenses/>.
END
gzip -9n > "$DOC/changelog.gz" <<END
$PACKAGE ($VERSION) unstable; urgency=medium

  * Release $VERSION. See $HOMEPAGE/releases.

 -- $MAINTAINER  $DATE
END

# Prebuilt third-party binaries: lintian's checks for Debian-built code do not apply.
cat > "$STAGE/usr/share/lintian/overrides/$PACKAGE" <<END
$PACKAGE: embedded-library *
$PACKAGE: unstripped-binary-or-object *
$PACKAGE: shared-library-lacks-prerequisites *
$PACKAGE: custom-library-search-path *
$PACKAGE: hardening-no-relro *
$PACKAGE: hardening-no-fortify-functions *
$PACKAGE: no-manual-page *
END

# Files only a developer needs, and permissions dpkg/lintian expect.
find "$LIB" -name '__pycache__' -prune -exec rm -rf {} +
find "$STAGE" -type d -exec chmod 755 {} +
find "$STAGE" -type f -exec chmod go-w,a+r {} +

INSTALLED=$(du -sk --apparent-size "$STAGE" | cut -f1)
cat > "$STAGE/DEBIAN/control" <<END
Package: $PACKAGE
Version: $VERSION
Architecture: $ARCH
Maintainer: $MAINTAINER
Installed-Size: $INSTALLED
Depends: libc6 (>= $GLIBC), libegl1, libgl1, libdrm2, libxcb1, libxcb-cursor0,
 libxcb-icccm4, libxcb-image0, libxcb-keysyms1, libxcb-render0, libxcb-render-util0,
 libxcb-shape0, libxcb-util1, libxrandr2, libwayland-client0, libwayland-cursor0,
 libwayland-egl1
Recommends: pulseaudio-utils
Conflicts: transifex-client
Section: sound
Priority: optional
Homepage: $HOMEPAGE
Description: local live audio transcription with speaker identification
 Transcribe turns speech from microphones, system audio (calls) or audio
 files into text, entirely on this computer. It labels who spoke, recognises
 known people again across sessions by their voiceprints, and exports
 transcripts as text, Markdown, SRT/VTT subtitles or JSON.
 .
 It provides the tx command and the Transcribe desktop app (tx-gui). Speech
 models are downloaded on first use.
END

mkdir -p "$OUT"
DEB=$OUT/${PACKAGE}_${VERSION}_${ARCH}.deb
dpkg-deb --root-owner-group -Zxz --build "$STAGE" "$DEB"
echo "deb: $DEB (glibc >= $GLIBC)"
