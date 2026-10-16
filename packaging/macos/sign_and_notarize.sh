#!/bin/sh
# Sign Transcribe.app (hardened runtime) and notarize it with Apple.
# Env: MACOS_CERT_P12 (base64), MACOS_CERT_PASSWORD, MACOS_SIGN_IDENTITY,
#      APPLE_ID, APPLE_TEAM_ID, APPLE_APP_PASSWORD (app-specific password)
set -eu
APP=$1
HERE=$(cd "$(dirname "$0")" && pwd)
KEYCHAIN=build.keychain
echo "$MACOS_CERT_P12" | base64 --decode > cert.p12
security create-keychain -p ci "$KEYCHAIN"
security default-keychain -s "$KEYCHAIN"
security unlock-keychain -p ci "$KEYCHAIN"
security import cert.p12 -k "$KEYCHAIN" -P "$MACOS_CERT_PASSWORD" -T /usr/bin/codesign
security set-key-partition-list -S apple-tool:,apple: -s -k ci "$KEYCHAIN"
rm cert.p12
# Sign nested code first (libraries, helper), then the app.
find "$APP/Contents" -type f \( -name "*.dylib" -o -name "*.so" -o -perm -u+x \) -print0 |
  xargs -0 codesign --force --options runtime --timestamp \
    --entitlements "$HERE/entitlements.plist" --sign "$MACOS_SIGN_IDENTITY"
codesign --force --options runtime --timestamp --entitlements "$HERE/entitlements.plist" \
  --sign "$MACOS_SIGN_IDENTITY" "$APP"
codesign --verify --deep --strict "$APP"
ditto -c -k --keepParent "$APP" notarize.zip
xcrun notarytool submit notarize.zip --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" \
  --password "$APPLE_APP_PASSWORD" --wait
xcrun stapler staple "$APP"
# Re-create the distributable archive with the stapled app.
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
rm -f dist/transcribe-*-macos-*.zip
ditto -c -k --keepParent "$APP" "dist/transcribe-$VERSION-macos-$(uname -m).zip"
