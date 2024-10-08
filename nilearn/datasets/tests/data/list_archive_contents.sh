#! /bin/bash

# generate a list of files and directories contained in a (zip or tar) archive
# downloaded from the network. Used to create the files found in
# /nilearn/datasets/tests/data/archive_contents. This script should be run on a
# unix system. The same result can easily be obtained manually: download the
# archive, list its contents (one item per line), using '/' as the path
# separator, add the url on the first line. See the docstring for
# nilearn.datasets._testing.tests.Sender for details.

url=$1
if [[ -z "$url" ]]; then
    echo "Usage: $0 URL" >&2
    exit 1
fi

tmpdir=$(mktemp -d)
cd "$tmpdir"

trap 'echo -e "\n\nDownload failed" >&2; exit 1' ERR
trap 'echo -e "\n\nDownload interrupted" >&2; exit 1' INT TERM
wget -O downloaded "$url" >&2
trap - ERR INT TERM


contents=$(2>/dev/null tar --list -f downloaded)
if [[ "$?" -eq 0 ]]; then
    echo "format: gztar"
    printf "%s\n" "$url"
    printf "%s\n" "$contents" | grep -v '/$'
    exit 0
fi

contents=$(2>/dev/null zipinfo -1 downloaded)
if [[ "$?" -eq 0 ]]; then
    echo "format: zip"
    printf "%s\n" "$url"
    printf "%s\n" "$contents" | grep -v '/$'
    exit 0
fi

echo "failed to list archive contents" >&2
exit 1
