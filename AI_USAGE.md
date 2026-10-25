# AI usage

## Tools I used, and for which functions

I used ChatGPT to understand the task requirements, clarify edge cases, and review implementation approaches for `audit()` in `rawaj/audit.py` and the metric functions in `rawaj/metrics.py`. I ran the provided tests using `uv run pytest`, and all tests passed.

## One place where the AI was wrong, incomplete, or not what the spec said

The AI initially focused on detecting target leakage through feature values but did not fully account for leakage caused by missingness patterns. I noticed this by reviewing the task clarifications, which require evaluating missingness indicators using ROC AUC as well. This helped me understand why both feature values and missingness patterns must be checked.

## The function I wrote without AI, and why

I independently implemented the constant-column detection logic within `audit()`. I wanted to understand the difference between constant-column detection and leakage screening. Constant-column detection ignores missing values when counting distinct values, whereas leakage screening treats missingness as a distinct value. This helped me understand why a column can be classified as constant while still being a potential leakage suspect.
