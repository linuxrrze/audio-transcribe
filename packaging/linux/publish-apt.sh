#!/bin/sh
# Write a signed flat apt repository index for .deb files into DIR.
#
#   packaging/linux/publish-apt.sh DIR PACKAGE.deb...
#
# The release workflow attaches DIR's files and the .debs to the GitHub Release, so
# https://github.com/OWNER/REPO/releases/latest/download/ is always an apt repository with
# the newest version (GitHub redirects each file to the latest release). DIR gets Packages,
# Release, InRelease, Release.gpg, the public keyring and transcribe.sources; the .debs
# themselves are not copied.
#
# Needs apt-ftparchive (apt-utils) and gpg with the secret signing key.
# APT_SIGNING_KEY_ID picks the key (default: the first secret key);
# APT_SIGNING_PASSPHRASE unlocks it if it has one.
set -eu

URL=${APT_REPO_URL:-https://github.com/linuxrrze/audio-transcribe/releases/latest/download/}
KEYRING=transcribe-archive-keyring

die() { echo "error: $*" >&2; exit 1; }
[ $# -ge 2 ] || die "usage: $0 DIR PACKAGE.deb..."
DIR=$1; shift
mkdir -p "$DIR"
command -v apt-ftparchive >/dev/null 2>&1 || die "apt-ftparchive not found (apt install apt-utils)"

KEY=${APT_SIGNING_KEY_ID:-$(gpg --batch --list-secret-keys --with-colons | awk -F: '/^fpr:/ { print $10; exit }')}
[ -n "$KEY" ] || die "no secret gpg key for signing"

sign() {  # sign OUT ARGS...: gpg with the repository key, the passphrase on stdin
	out=$1; shift
	if [ -n "${APT_SIGNING_PASSPHRASE:-}" ]; then
		printf '%s' "$APT_SIGNING_PASSPHRASE" | gpg --batch --yes --pinentry-mode loopback \
			--passphrase-fd 0 --local-user "$KEY" --output "$out" "$@"
	else
		gpg --batch --yes --local-user "$KEY" --output "$out" "$@"
	fi
}

# apt-ftparchive names each file by its path, so scan the .debs from a directory that
# holds only them: Filename becomes ./NAME.deb, next to the index as in the release.
POOL=$(mktemp -d)
trap 'rm -rf "$POOL"' EXIT
for deb in "$@"; do
	case $deb in *.deb) ;; *) die "not a .deb: $deb" ;; esac
	[ -f "$deb" ] || die "$deb not found"
	case $(basename "$deb") in *[!A-Za-z0-9._+-]*) die "GitHub renames this asset: $deb" ;; esac
	ln -s "$(cd "$(dirname "$deb")" && pwd)/$(basename "$deb")" "$POOL/"
done
(cd "$POOL" && apt-ftparchive packages .) > "$DIR/Packages"
grep -q '^Package:' "$DIR/Packages" || die "no packages in the index"
gzip -9n -k -f "$DIR/Packages"
xz -k -f "$DIR/Packages"

cd "$DIR"
rm -f Release InRelease Release.gpg
apt-ftparchive \
	-o APT::FTPArchive::Release::Origin="Transcribe" \
	-o APT::FTPArchive::Release::Label="Transcribe" \
	-o APT::FTPArchive::Release::Description="Transcribe packages for Debian and Ubuntu" \
	release . > Release.tmp
mv Release.tmp Release
sign InRelease --clearsign Release
sign Release.gpg --armor --detach-sign Release

gpg --batch --yes --output $KEYRING.gpg --export "$KEY"
gpg --batch --yes --armor --output $KEYRING.asc --export "$KEY"
cat > transcribe.sources <<END
Types: deb
URIs: $URL
Suites: ./
Signed-By: /usr/share/keyrings/$KEYRING.gpg
END
echo "apt index in $DIR, signed by $KEY:"
sed -n 's/^Filename: /  /p' Packages
