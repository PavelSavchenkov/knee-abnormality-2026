# Agent Instructions

These instructions apply unless explicitly overridden.

## Code 

- Keep code concise and names readable; follow KISS and Occam's razor.
- In Python, use direct imports; no `try`/`except` import fallbacks.
- Fail fast on unexpected states; add fallbacks only when strongly justified.
- Omit `-> None` from functions that return nothing.
- Do not add `__init__.py` files.
- Do not reinvent standard and common logic, look for suitable packages and libraries first

## Language in docs and chat

- When asked for markdown in chat interface, provide it as unrendered plain text.
- Use the simplest concrete wording that preserves the full meaning. Do not add abstraction, jargon, or explanation unless it improves precision or avoids repetition.
- Prefer direct, practical sentences. Remove wording that does not change a fact, decision, requirement, metric, or action.
- Example: replace “The blind novelty reserve remains unprofiled” with “Keep some opponents unused and unanalyzed until the final agent is ready.”

## General

- Do not use subagents.
- Run all Python, Kaggle, notebook, and package-management commands via `conda run -n rsna <command>`.
- You can install common packages into "rsna" conda env when needed. Prefer using mamba, then conda, then pip.
- Do not use Git.
- Create new `experiments/` subfolders only when explicitly asked.
- Store submission-related artifacts in `submissions/<date>_<name>/`.
- Store WIP scripts and artifacts in `work/` by default. Do not pollute repo root and unrelated folders.
- Keep `experiments/<version>/<date>_<name>/` self-contained: store all experiment artifacts there and never reference code from another experiment (if you need to borrow the code, just copy artifacts from other experiments).
- Use host provided `data/original_data/test.csv`+`test_series.csv`+`sample_submission.csv` only as a format example; it is unrelated to public or private tests.
- This is a code competition. Official kaggle submissions accept notebooks which read private test only visible in scoring time and produce its predictions. We do not submit any CSVs.
- For any official kaggle submission, follow `prompts/submission.md`
- Follow `prompts/validation.md`
- When asked to set up a regular review (typically with wake up hook) follow `prompts/regular_review.md` by default.
