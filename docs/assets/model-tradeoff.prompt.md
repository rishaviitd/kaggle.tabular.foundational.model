# Model comparison infographic prompt

Tool: built-in image generation, edit mode.
Reference: docs/assets/model-tradeoff.png from Git commit 9d4446e,
replaced after verified tuning and submission.

Use case: infographic-diagram (edit)
Edit the provided housing-affordability comparison infographic. Preserve its polished navy/ivory editorial style, blue XGBoost column, teal TabPFN column, CPU/GPU illustrations and housing/map motif. Reflow vertically with a shorter header illustration band to fit FIVE comparison rows and readable footnotes. Keep all text sharp and numbers exact. This is the final verified version after XGBoost tuning and a new same-host benchmark.
Title remains "Accuracy vs inference speed".
Subtitle: "Housing affordability · tuned XGBoost CPU vs optimized TabPFN".
Column headings: "Tuned XGBoost" / "4 CPU threads" and "Optimized TabPFN v2" / "1 × Tesla T4".
Replace the entire metrics grid with exactly these FIVE rows, XGBoost value first then TabPFN:
"Single-request latency ↓" : "77 ms" | "145 ms"
"Throughput ↑" : "701 req/s" | "164 req/s"
"Local RMSE · seed 42 ↓" : "5.681" | "4.623"
"Local RMSE · seed 137 ↓" : "5.545" | "5.117"
"Kaggle public score ↓" : "5.225*" | "3.919*"
Highlight XGBoost latency and throughput as blue winners. Highlight TabPFN BOTH local RMSE values AND Kaggle score as teal winners. No "Not submitted" text anywhere; both models now have verified submissions.
Preserve the two compact pros/cons cards with these headings and exact copy:
"Tuned XGBoost": "+ Faster CPU serving"; "− Higher local prediction error".
"Optimized TabPFN v2": "+ Lower local prediction error"; "− GPU required for this setup".
Replace footer with these FOUR readable lines:
"Same Kaggle host · unseen ZIPs · localhost HTTP · three trials"
"Latency: one client. Throughput: 256 clients; CPU uses a larger batching window."
"Local RMSE: 1,366 held-out ZIPs per split; distinct from the Kaggle metric."
"* Both use external data. TabPFN public score uses the original uncached model."
No extra numbers, no invented metrics, no graphs with misleading axes. All exact text must fit without clipping. This is a precise data infographic, not a stylized poster with illegible fine print. Landscape format, balanced spacing, high resolution.
