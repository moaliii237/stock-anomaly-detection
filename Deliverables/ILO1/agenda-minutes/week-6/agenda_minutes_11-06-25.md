# Agenda and Minutes *11/06/25*

## Meeting Agenda - *Team 13* - *11/06/25*

**Group Number**: 13
**Date**: 11/06/25
**Time**: 20:00-20:30
**Location**: Teams Call
**Chair**: Szymon Chirowski (242621)
**Minutes Taker**: Szymon Chirowski (242621)
**Attendees**:

* Szymon Chirowski (242621)
* Hristiyan Georgiev (241226)
* Mohammadali Jaberi (244437)
* Aristotelis Protopapas (234062)

---

### Agenda Items

| # | Topic                                     | Presenter   | Duration | Notes                                                 |
| - | ----------------------------------------- | ----------- | -------- | ----------------------------------------------------- |
| 1 | UI and deployment discussion              | Hristiyan   | 10 min   | UI stack, deployment integration with ML pipeline     |
| 2 | Final ML pipeline and modularization plan | Szymon      | 10 min   | Project structure and API interface                   |
| 3 | Hyperparameter tuning & code formatting   | Mohammadali | 10 min   | Prioritizing tuning over new model, logging standards |

---

## Meeting Minutes - *Team 13* - *11/06/25*

**Group Number**: 13
**Date**: 11/06/25
**Time**: 20:00-20:50
**Location**: Teams Call
**Chair**: Szymon Chirowski (242621)
**Minutes Taker**: Szymon Chirowski (242621)
**Attendees**:

* Szymon Chirowski (242621) - Present
* Hristiyan Georgiev (241226) - Present
* Mohammadali Jaberi (244437) - Present
* Aristotelis Protopapas (234062) - Absent (No show, no prior notice)

---

### Discussion Summary

#### Agenda item 1: UI and deployment discussion

##### Key Discussions

* Hristiyan will deliver a functional dashboard by Friday using Flask and HTML.
* UI will connect to saved models for inference through an endpoint.
* Szymon and Hristiyan discussed the integration of prediction logic through Flask routing.
* Clarified UI doesn't need to support model training, only prediction.

##### Decisions Made

* Hristiyan will continue with Flask due to prior experience.
* Focus will remain on deploying working prediction flow rather than visual polish.

---

#### Agenda item 2: Final ML pipeline and modularization plan

##### Key Discussions

* Szymon proposed organizing the repo into 3-4 main modules: preprocessing, modeling, dashboard, and optionally training.
* Manual prediction scripts will be integrated through endpoint calls.
* Confirmed only one final model needs to be included for submission.

##### Decisions Made

* Repo structure will include clearly separated modules for preprocessing, training, and prediction.
* Final model and artifacts will be included as static files.
* Model training logic will not be exposed in the dashboard interface.

---

#### Agenda item 3: Hyperparameter tuning & code formatting

##### Key Discussions

* Team decided to focus on hyperparameter tuning rather than adding a new model (TFT).
* Mohammadali will use manual search or cross-validation for tuning.
* Logging will replace print statements for MLOps formatting compliance.

##### Decisions Made

* Mohammadali to perform hyperparameter tuning and document progress in Trello.
* Logging setup will follow template configuration provided by Szymon.
* Codebase will be formatted with Black; front-end is excluded from formatting standards.

---

### Action Items

| Task                                               | Responsible  | Deadline |
| -------------------------------------------------- | ------------ | -------- |
| Finalize UI and prediction endpoint with Flask     | Hristiyan    | 14/06/25 |
| Hyperparameter tuning using manual or CV methods   | Mohammadali  | 14/06/25 |
| Repo modularization (training, prediction)     | Szymon       | 14/06/25 |
| Apply Black formatting and logging configuration   | All | 14/06/25 |
| Add poetry environment setup and screenshots       | All          | 14/06/25 |
| Start README draft (structure, decisions, results) | All          | 17/06/25 |

> *Make sure all tasks and deadlines are also logged in Trello or your task manager.*

---

### Next Meeting

**Date**: 12/06/25
**Time**: 13:00-13:15
**Location**: Teams Call
**Chair**: Szymon Chirowski (242621)
**Minutes Taker**: Szymon Chirowski (242621)

---
