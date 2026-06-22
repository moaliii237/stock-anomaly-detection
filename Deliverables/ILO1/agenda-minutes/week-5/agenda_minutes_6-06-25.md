# Agenda and Minutes *06/06/2025*

## Meeting Agenda - *Team 13* - *06/06/2025*

**Group Number**: 13
**Date**: `06/06/2025`
**Time**: `10:00 - 10:30`
**Location**: `On Campus and Teams Call`
**Chair**: `Szymon Chirowski (242621)`
**Minutes Taker**: `Szymon Chirowski (242621)`
**Attendees**:

* `Szymon Chirowski (242621)`
* `Hristiyan Georgiev (241226)`
* `Mohammadali Jaberi (244437)`
* `Aristotelis Protopapas (234062)` - Online (train strike)
* `Elavendan Rajendran`

---

### Agenda Items

| # | Topic                                              | Presenter            | Duration | Notes                                                   |
| - | -------------------------------------------------- | -------------------- | -------- | ------------------------------------------------------- |
| 1 | Progress on modeling tasks and hybrid model        | Szymon, Mohammadali, Hristiyan | 10 min   | Updates on LSTM, hybrid CNN models, and RF/DT           |
| 2 | CNN model progress & UI status update              | Aristotelis          | 10 min   | Clarification of CNN progress and dashboard alignment   |
| 3 | Clarifications on SQL tasks and template questions | Szymon, Elavendan             | 10 min   | Addressing doubts about SQL security and final notebook |

---

## Meeting Minutes - *Team 13* - *06/06/2025*

**Group Number**: 13
**Date**: `06/06/2025`
**Time**: `10:00 - 10:30`
**Location**: `On Campus and Teams Call`
**Chair**: `Szymon Chirowski (242621)`
**Minutes Taker**: `Szymon Chirowski (242621)`
**Attendees**:

* `Szymon Chirowski (242621)` - Present
* `Hristiyan Georgiev (241226)` - Present
* `Mohammadali Jaberi (244437)` - Present
* `Aristotelis Protopapas (234062)` - Present
* `Elavendan Rajendran` - Present

---

### Discussion Summary

#### Agenda item 1: Progress on modeling tasks and hybrid model

##### Key Discussions - agenda item 1

* Hristiyan shared progress on the hybrid CNN-LSTM model.
* Szymon showed his progress with LSTM model, initial runs were time-consuming due to environment issues, but promising (7% validation accuracy vs 0% baseline).
* Mohammadali reported success with RF and DT models using class weighting; best F1 score \~30%.
* Discussion on preprocessing: Hristiyan clarified that main preprocessing (target shift, scaling) was already completed and shared in the repo.

##### Decisions Made - agenda item 1

* Szymon to continue hybrid model optimization and address environment issues.
* Mohammadali to finalize performance comparisons for all models.
* Team agreed to prioritize model performance results before week 7.

---

#### Agenda item 2: CNN model progress & UI status update

##### Key Discussions - agenda item 2

* Aristotelis reported limited progress: only basic preprocessing done despite team preprocessing already shared.
* Dashboard demo showed 3 pages with dummy data; concerns raised about relevance of the charts and unclear linkage to project goals.
* Team and mentor emphasized urgent need for Aristotelis to align with project objectives, integrate team models into dashboard, and stop pursuing unrelated ideas (live API).
* Mentor issued a second formal warning regarding contribution quality and deadline risk (possible group contribution = 0 if no meaningful work by next Tuesday).

##### Decisions Made - agenda item 2

* Aristotelis must:

  * Complete CNN model and show results.
  * Refactor dashboard to match project scope and integrate proper predictions.
  * Follow Git workflow (branch + PR) and contribute properly to the repo.
* Other members may take over UI if no progress is made by next week.

##### Unresolved Issues

* Dashboard currently not usable as-is; urgent rework needed.
* CNN model not yet delivered.

---

#### Agenda item 3: Clarifications on SQL tasks and template questions

##### Key Discussions - agenda item 3

* Szymon asked about "Create table for queries" - clarified this refers to saving query results in a new table.
* Security questions: no need to implement; just document role-based access suggestions in Markdown.
* Template questions: how to document preprocessing (OK to call existing functions), what is expected in final step (model output for UI).

##### Decisions Made - agenda item 3

* Szymon to proceed as discussed and update To Do list accordingly.
* No blockers identified for SQL work or template completion.

---

### Action Items

| Task                                                         | Responsible | Deadline   |
| ------------------------------------------------------------ | ----------- | ---------- |
| Continue optimizing hybrid CNN-LSTM model                    | Szymon      | Ongoing    |
| Finalize RF/DT/XGB model performance results                 | Mohammadali | Ongoing    |
| Implement CNN model and refactor dashboard per team feedback | Aristotelis | 11-06-2025 |
| Review and sign off peer-reviewed SQL and EDA                | All         | ASAP       |

> *Make sure all tasks and deadlines are also logged in Trello or your task manager.*

---

### Next Meeting

**Date**: `10/06/2025`
**Time**: `10:00 - 10:30`
**Location**: `On Campus and Teams Call`
**Chair**: `Szymon Chirowski (242621)`
**Minutes Taker**: `Szymon Chirowski (242621)`
