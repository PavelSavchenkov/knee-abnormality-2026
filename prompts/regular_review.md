# Regular review

Unless the user says otherwise, do these at every regular review (typically every 10 minutes), where they apply. Write a short review
entry in the session's log: what changed and what you decided.

## 1. Progress and direction

- Look at the whole progress since the session started, not only the last 10 minutes. Are we making good progress toward the goal?
- Decide whether to pivot.

## 2. What is running

- Check that every run does what we intended: config, data, labels, folds, seeds, flags, checkpoint / epoch.
- Check for crashed, stuck or slow runs and dead monitors; fix or stop them.
- Owned GPU / CPU must not sit idle: keep the queue filled, or release the GPU at once.
- Audit the GPU queues line by line: each pending line serves a named decision, with its expected gain and minutes; the best gain per
  minute runs first, fillers last; fix the queue in the same review (Master: MASTER_CHECKLIST E2, work/queue_audit/).
- After EVERY GPU run (each run, each fold), not only at the review: read it, then re-prioritise the queue and the backlog
  (USER 10-20: "you always must do reprioritisation after each gpu run"). Do not queue bundled 3-fold trainings: one fold per arm, the
  next fold only when the read so far justifies it and it outranks unscreened arms (USER 10-20: "do reprioritisation after each run and
  each fold, do not do bundle 3 folds trainings").

## 3. Code and data

- Read the code line by line. Look for bugs and inconsistencies.
- Ask whether we are testing what we wanted to test.
- Check that the data is clean and as expected, and that the code is correct and fast enough.

## 4. Results and validation

- Read every new result and log it with its evidence strength (model, folds, seeds).
- Check that the validation is as fair and useful as it can be (prompts/validation.md). Read each result in the form it will be
  deployed in.
- Update conclusions; correct earlier ones that new evidence contradicts.

## 5. Ideas

- Re-prioritise the ideas ledger by:
  - estimated ROI;
  - how underexplored the idea is;
  - how fast and cheap it is to try;
  - simplicity;
  - how evidence-based it is.
- Spread ideas across classes and levers. Do not keep iterating on the same one to three classes.
- A failed idea means one specific recipe (hparams, data, all other settings) failed under a given validation setting and noise scale.
  The direction is not closed; rank it by how underexplored it still is.
- Add new ideas from:
  - this session's learnings;
  - other experiment folders and sessions;
  - public discussions;
  - similar past Kaggle competitions;
  - the literature;
  - general ML / DL practice.
- Adapt ideas that failed elsewhere when the setting differs.
- Order screens and full runs to get the most useful information fastest. Follow promising ideas deeply enough before moving on.

## 6. People and probes

- Log every user suggestion since the last review, and say what was done about it.
- Check messages from other sessions. Ask them for context, and send them findings that concern them.
- Check Kaggle probe status. Propose probes for results that stay uncertain locally (prompts/submission.md; every submission needs
  the user's approval).

## While waiting (no GPU, nothing to read)

Do an extended review:
- study past experiments in depth;
- re-check the code;
- prepare the next runs;
- run useful CPU checks.
