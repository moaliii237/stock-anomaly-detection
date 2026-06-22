# Agenda and Minutes *28-05-2025*

## Meeting Agenda - *Team 13* - *28-05-2025*

**Group Number**: 13
**Date**: `28-05-2025`
**Time**: `17:00 - 17:30`
**Location**: `Teams Call`
**Chair**: `Szymon Chirowski (242621)`
**Minutes Taker**: `Szymon Chirowski (242621)`
**Attendees**:

* `Szymon Chirowski (242621)`
* `Hristiyan Georgiev (241226)`
* `Mohammadali Jaberi (244437)`
* `Aristotelis Protopapas (234062)`

---

### Agenda Items

| # | Topic                                                                    | Presenter   | Duration | Notes                              |
| - | ------------------------------------------------------------------------ | ----------- | -------- | ---------------------------------- |
| 1 | Data exploration and processing conclusions: what we have, what we don't | Hristiyan   | 10 min   | Overview of current state of data  |
| 2 | Discussion on initiating the first model builds                          | Szymon      | 10 min   | Assess readiness to start modeling |
| 3 | Task division for the upcoming week                                      | Team        | 5 min    | Clarify who does what next         |
| 4 | Progress update on Aristotelis' task                                     | Aristotelis | 5 min    | UX/UI research task follow-up      |

---

## Meeting Minutes - *Team 13* - *28-05-2025*

**Group Number**: 13
**Date**: `28-05-2025`
**Time**: `17:00 - 17:15`
**Location**: `Teams Call`
**Chair**: `Szymon Chirowski (242621)`
**Minutes Taker**: `Szymon Chirowski (242621)`
**Attendees**:

* `Szymon Chirowski (242621)` - Present
* `Hristiyan Georgiev (241226)` - Present
* `Mohammadali Jaberi (244437)` - Present
* `Aristotelis Protopapas (234062)` - Present

---

### Discussion Summary

#### Agenda item 1: Data exploration and processing conclusions

##### Key Discussions

* Hristiyan described the current dataset combining minute-by-minute and daily Apple stock data.
* Excel format is used due to CSV formatting issues.
* Grouping events by 30-minute intervals is planned, but shorter intervals (15, 10, 5 mins) will also be explored.
* Awaiting modeling feedback before integrating external datasets.

##### Decisions Made

* Continue working with Excel for now.
* Delay merging external datasets until current dataset is validated.
* Test multiple time window granularities for event prediction.

##### Unresolved Issues

* Conversion of dataset from Excel to properly formatted CSV without corrupting number formats.
* Final decision on optimal time window for event grouping.

---

#### Agenda item 2: Discussion on initiating the first model builds

##### Key Discussions

* Mohammadali began working on a random classifier and is reviewing dataset structure.
* Planning to experiment with short-term memory (LSTM) models.
* Szymon will try basic classifiers for comparison.
* Discussion about limiting the scope to Apple stock for initial model validation.

##### Decisions Made

* Continue with both LSTM and traditional classifiers.
* Share performance metrics and models for peer review.
* Focus solely on Apple stock initially for proof-of-concept modeling.

##### Unresolved Issues

* Final model approach still undecided; depends on performance results.

---

#### Agenda item 3: Task division for the upcoming week

##### Key Discussions

* Mohammadali will document planned modeling tasks in Trello.
* Hristiyan will delay further data engineering until feedback is received but is available to help with model execution.
* Szymon will test simpler models and provide feedback.

##### Decisions Made

* Update Trello with all individual responsibilities and task statuses.
* Cross-support between modeling and data processing if needed.

##### Unresolved Issues

* Clarity on next dataset engineering steps pending modeling results.

---

#### Agenda item 4: Progress update on Aristotelis' task

##### Key Discussions

* Aristotelis created a draft document detailing UI/UX best practices and technology stack.
* The draft will be shared for team review via Teams.
* Estimated final version by Monday, possibly sooner.

##### Decisions Made

* Upload the draft to Teams for review.
* Submit final version by 02-06-2025.

##### Unresolved Issues

* Pending team feedback on the document.
* Final adjustments may be required based on review.

---

### Action Items

| Task                                                    | Responsible | Deadline   |
| ------------------------------------------------------- | ----------- | ---------- |
| Upload initial modeling results and performance summary | Mohammadali | 29-05-2025 |
| Share draft UI/UX stack and design guideline doc        | Aristotelis | 30-05-2025 |
| Review and provide feedback on UI/UX doc                | All         | 30-05-2025 |
| Experiment with classic classifiers as baseline         | Szymon      | 03-06-2025 |
| Log individual tasks and updates in Trello              | All         | Ongoing    |

> *Make sure all tasks and deadlines are also logged in Trello or your task manager.*

---

### Next Meeting

**Date**: `02-06-2025`
**Time**: `TBD`
**Location**: `Teams Call`
**Chair**: `Szymon Chirowski (242621)`
**Minutes Taker**: `Szymon Chirowski (242621)`
