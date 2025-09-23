#!/usr/bin/env bash
set -euo pipefail

PKG="nilearn"
OUTDIR="nilearn_sizes"
CSV="sizes.csv"

mkdir -p "$OUTDIR"
echo "version,file,size_bytes" > "$CSV"

# 1. Get all available versions from PyPI
VERSIONS=$(curl -s "https://pypi.org/pypi/${PKG}/json" | jq -r '.releases | keys[]' | sort -V)

for v in $VERSIONS; do
    echo "Processing $PKG $v ..."
    # Get the URL of a wheel if available, otherwise fall back to sdist
    URL=$(curl -s "https://pypi.org/pypi/${PKG}/${v}/json" | jq -r '
        .urls[] | select(.packagetype == "bdist_wheel") | .url
    ' | head -n1)

    if [ -z "$URL" ]; then
        # fallback to sdist
        URL=$(curl -s "https://pypi.org/pypi/${PKG}/${v}/json" | jq -r '
            .urls[] | select(.packagetype == "sdist") | .url
        ' | head -n1)
    fi

    if [ -z "$URL" ]; then
        echo "⚠️ No artifact found for $v"
        continue
    fi

    FILE="$OUTDIR/$(basename "$URL")"
    if [ ! -f "$FILE" ]; then
        curl -sL "$URL" -o "$FILE"
    fi

    SIZE=$(stat -c%s "$FILE" 2>/dev/null || stat -f%z "$FILE")
    echo "$v,$(basename "$FILE"),$SIZE" >> "$CSV"
done

# 2. Generate plot using Python
python3 <<'EOF'
import pandas as pd
import matplotlib.pyplot as plt

from packaging.version import Version

df = pd.read_csv("sizes.csv")
df["parsed_version"] = df["version"].apply(Version)
df = df.sort_values("parsed_version")

plt.figure(figsize=(10,5))
plt.plot(df["version"], df["size_bytes"]/1e6, marker="o")
plt.xticks(rotation=90)
plt.ylabel("Package size (MB)")
plt.xlabel("Nilearn version")
plt.title("Evolution of Nilearn package size over versions")
plt.tight_layout()
plt.grid(True)
plt.savefig("nilearn_size_evolution.png", dpi=150)
print("✅ Plot saved to nilearn_size_evolution.png")
EOF
