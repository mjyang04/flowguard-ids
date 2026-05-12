#!/usr/bin/env bash
# Recompile the thesis from the 7 chapter Markdown sources.
#
# Usage:
#   bash docs/thesis/compile.sh          # produces thesis.md + thesis.docx
#   bash docs/thesis/compile.sh --md     # Markdown only
#   bash docs/thesis/compile.sh --docx   # .docx only

set -euo pipefail
cd "$(dirname "$0")"

chapters=(
  front_matter.md
  01_introduction.md
  02_related_work.md
  03_research_methodology.md
  04_results_and_discussion.md
  05_conclusion.md
  references.md
)

mode="${1:-both}"

if [[ "$mode" == "both" || "$mode" == "--md" ]]; then
  echo "[compile] merging ${#chapters[@]} chapters into thesis.md"
  cat "${chapters[@]}" > thesis.md
  echo "[compile] thesis.md -> $(wc -w < thesis.md) words"
fi

if [[ "$mode" == "both" || "$mode" == "--docx" ]]; then
  echo "[compile] invoking pandoc with FYP reference template"
  pandoc "${chapters[@]}" \
    --reference-doc="FYP Thesis Template  042025 v2.docx" \
    -o thesis.docx
  echo "[compile] thesis.docx -> $(ls -la thesis.docx | awk '{print $5}') bytes"
fi

echo "[compile] done."
