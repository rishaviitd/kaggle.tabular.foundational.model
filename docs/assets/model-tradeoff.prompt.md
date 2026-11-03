# Model comparison infographic prompt

Tool: built-in image generation. No input images.

Use case: infographic-diagram
Asset type: polished landscape PNG for a GitHub machine-learning repository.
Create a beautiful, precise editorial infographic titled "Accuracy vs inference speed" with subtitle "Housing affordability · XGBoost CPU vs optimized TabPFN".
Dark navy background, crisp ivory typography, restrained blue for XGBoost and teal for TabPFN, generous spacing, elegant CPU and GPU line illustrations, no official logos. A clean comparison grid with row labels and two model columns. Large perfectly legible numbers, no decorative random text. Landscape 3:2 composition, high resolution.
Column headings: "XGBoost" / "4 CPU threads" and "Optimized TabPFN v2" / "1 × Tesla T4".
Exactly these metric rows:
"Single-request latency ↓" with "78 ms" under XGBoost and "148 ms" under TabPFN.
"Throughput ↑" with "731 req/s" under XGBoost and "156 req/s" under TabPFN.
"Local RMSE ↓" with "5.737" under XGBoost and "4.623" under TabPFN.
"Kaggle public score ↓" with "Not submitted" under XGBoost and "3.919*" under TabPFN.
Use tiny stopwatch, request stack, target and leaderboard icons in row labels. Highlight latency and throughput XGBoost winner numbers in blue; local RMSE TabPFN winner in teal. No winner claim for Kaggle since XGBoost has no score. Use number cards, not charts with axes.
Below the grid, two compact balanced pros/cons cards:
XGBoost: "+ Faster CPU serving" and "− Higher local prediction error".
TabPFN: "+ Lower local prediction error" and "− GPU required for this setup".
Readable footer, exact text:
"Same Kaggle host · unseen ZIPs · localhost HTTP · three trials"
"Latency: one client. Throughput: 256 clients; CPU uses a larger batching window."
"Local RMSE: seed 42, 1,366 held-out ZIPs; distinct from the Kaggle metric."
"* External-data research submission. Public score uses the original uncached model."
Keep all digits and labels exact. Clear separation of local RMSE and public Kaggle score. No claim these finite tests are maximum capacity; no invented dollar cost, no accuracy percentages. Sophisticated research figure design with understated housing/ZIP-map geometric motif only in header.
