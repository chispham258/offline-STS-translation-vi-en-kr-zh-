#!/usr/bin/env bash
# Render docs/diagrams/*.mmd to SVG. Source .mmd files are the source of truth.
# Requires Node/npx; mermaid-cli downloads Chromium on first run.
set -euo pipefail

cd "$(dirname "$0")/.."

command -v npx >/dev/null 2>&1 || {
  echo "npx not found; install Node.js first" >&2
  exit 1
}

mkdir -p docs/diagrams

for src in docs/diagrams/*.mmd; do
  out="docs/diagrams/$(basename "${src%.mmd}").svg"
  echo "render $src -> $out"
  npx -y @mermaid-js/mermaid-cli@12.0.0 -i "$src" -o "$out" -b white
done
