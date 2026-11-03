# Model comparison hero image prompt

Tool: built-in image generation, edit mode.
Reference: docs/assets/model-tradeoff.png from Git commit 8c0fb18.

Use case: infographic-diagram, edit.
Simplify this supplied infographic into a spacious, attractive GitHub README HERO IMAGE. Preserve its warm ivory paper background, walnut-brown headings, terracotta CPU accents, forest-green GPU accents, housing illustration and tasteful CPU/GPU icons. Clean, bright editorial design, large readable numbers and generous whitespace. Landscape 3:2. Make it feel like the opening visual of a professional ML repository, not a dense benchmark report.

Title: "Accuracy vs inference speed"
Subtitle: "Housing affordability · XGBoost vs TabPFN"

Model column headers:
"XGBoost" with smaller "CPU · 4 threads"
"TabPFN v2" with smaller "GPU · 1 × Tesla T4"
Keep attractive small hardware icons. Remove the word "Tuned" EVERYWHERE. Remove tree counts, depth, estimator counts, Python package versions, and all dense header metadata; those live in the docs.

Exactly FIVE aligned metric rows, XGBoost left then TabPFN right:
"Single-request latency ↓" : "77 ms" | "145 ms"
"Throughput ↑" : "701 req/s" | "164 req/s"
"Model file size" : "3.94 MB" | "44.39 MB"
"Local RMSE ↓" : "5.613" | "4.870"
"Kaggle public score ↓" : "5.225" | "3.919"

IMPORTANT Local RMSE values now represent the arithmetic mean of the two previously reported split RMSEs; the image shows ONE local score per model. No seeds, no second local row, no split labels. Keep this method explanation OUT of the image (it goes in documentation).
Highlight XGBoost latency and throughput subtly in terracotta, and TabPFN local/public error subtly in forest green. Keep model-size values neutral.

Below the table, retain only two light compact takeaway cards:
under XGBoost: "Faster CPU inference"
under TabPFN: "Lower prediction error"
No extra bullet lists or lengthy pros/cons.

REMOVE THE ENTIRE OLD FOOTER. No footer text, no asterisks, no footnotes, no methodology, no eligibility text, no fine print, no bottom divider suggesting missing text. End with clean breathing room below the two takeaway cards. Detailed measurement context and caveats are in the README/docs, not this visual.
All five metric rows and exact values must be preserved without invention. Do not shrink type to fill the old space; use removed text space for generous spacing and better hierarchy. No bar charts, no noisy wood grain.
