# Model comparison infographic prompt

Tool: built-in image generation, edit mode.
Reference: docs/assets/model-tradeoff.png from Git commit 6394cfe.

Use case: infographic-diagram, edit.
Edit the supplied comparison PNG into a substantially more spacious, brighter, polished report infographic. This is a redesign, not just a recolor. Use a roomy 4:3 landscape canvas, ideally 2400×1800, with generous outer margins and generous row padding. Warm ivory/paper background, walnut wood-brown headings, tasteful terracotta accents for CPU and deep forest/teal accents for GPU. Subtle warm natural tones, NO busy woodgrain behind text, NO dark navy background, no neon/glowing outlines. Modern readable sans-serif for data, elegant editorial title, thin dividers, plenty of whitespace. Simplify CPU/GPU graphics into attractive clean recognizable hardware illustrations with warmth, no noisy oversized gaming graphics. Tiny housing motif if useful. Keep logical aligned comparison columns and readable footnotes.

Exact title: "Accuracy vs inference speed"
Subtitle: "Housing affordability · same features, different tradeoffs"

Two model header cards, aligned over metric columns:
LEFT: "Tuned XGBoost", "CPU · 4 threads", "1,462 trees · depth 5", "XGBoost 3.4.1".
RIGHT: "TabPFN v2", "GPU · 1 × Tesla T4", "4 estimators", "Python package: tabpfn 9.2.0".
Clearly distinguish TabPFN MODEL version v2 from Python PACKAGE version 9.2.0. No model called v9.

Six clear aligned metric rows with exact text and values, LEFT XGBoost then RIGHT TabPFN:
"Single-request latency ↓" : "77 ms" | "145 ms"
"Throughput ↑" : "701 req/s" | "164 req/s"
"Model file size" : "3.94 MB" | "44.39 MB"
"Local RMSE · seed 42 ↓" : "5.681" | "4.623"
"Local RMSE · seed 137 ↓" : "5.545" | "5.117"
"Kaggle public score ↓" : "5.225*" | "3.919*"
File-size right value is the shared base checkpoint size, NOT four checkpoint copies or runtime memory. Label small beneath size row "Saved XGBoost model / TabPFN base checkpoint". Highlight CPU speed wins subtly in terracotta and TabPFN error wins subtly in teal. Lower error and latency are better; higher throughput is better. No fake bar scales.

Two compact spacious tradeoff summaries below:
"Tuned XGBoost": "+ Faster CPU serving", "− Higher prediction error".
"TabPFN v2": "+ Lower prediction error", "− GPU used in this setup".

Required legible footnotes, placed with breathing space at bottom, exact content:
"Same Kaggle host · unseen ZIPs · localhost HTTP · three trials"
"Latency: one client. Throughput: 256 clients; CPU uses a larger batching window."
"Local RMSE: 1,366 held-out ZIPs per split; distinct from the Kaggle metric."
"File sizes use decimal MB; runtime RAM/VRAM and TabPFN context/cache are extra."
"* Both use external data. TabPFN public score uses the original uncached model."

All measurements and digits must remain exact. No omitted metric, no invented information, no cramped text, no clipping. Prioritize readable balanced visual hierarchy over decorative hardware images. Add whitespace by expanding the canvas rather than shrinking type.
