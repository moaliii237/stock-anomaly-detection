# Agenda and Minutes *04-06-2025*

## Meeting Agenda - *Team 13* - *04-06-2025*

**Group Number**: 13
**Date**: `04/06/2025`
**Time**: `13:30 - 13:45`
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

| # | Topic                                    | Presenter   | Duration | Notes                               |
| - | ---------------------------------------- | ----------- | -------- | ----------------------------------- |
| 1 | Review of Szymon's EDA                   | Szymon      | 5 min   | Summarization & feedback needed     |
| 2 | Modeling Responsibilities Clarification  | All         | 2 min   | Confirm tasks and split models      |
| 3 | Dashboard Progress & Demo                | Aristotelis | 5 min   | Live demo and design discussion     |
| 4 | Resampling Strategy and Dataset Versions | Szymon      | 1 min   | -   |
| 5 | Task planning for next 24h               | All         | 1 min    | Clarifying work before next standup |

---

## Meeting Minutes - *Team 13* - *04/06/2025*

**Group Number**: 13
**Date**: `04/06/2025`
**Time**: `13:30 - 13:45`
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

#### Agenda item 1: Review of Szymon's EDA

##### Key Discussions

* Szymon asked for feedback on the EDA shared via GitHub; Hristiyan reviewed it, but Mohammadali hadn't checked it yet.
* The EDA was confirmed to be solid, but a summarized interpretation of insights was requested.

##### Decisions Made

* Szymon will provide a summary of the EDA findings.
* Team members will review EDA on GitHub and provide comments.

---

#### Agenda item 2: Modeling Responsibilities Clarification

##### Key Discussions

* Szymon will work on LSTM, SVM, and TFT models, as these are complex and training-intensive.
* Mohammadali confirmed he's currently working on XGBoost, Random Forest, and Decision Tree models.
* Aristotelis was assigned CNN model for time series.
* Hristiyan will work on the LSTM model as it is known to be the most time and resource-consuming, so he will assist Szymon with it.
* Each team member must document their assigned models on Trello.

##### Decisions Made

* Modeling assignments were confirmed:

  * Szymon: LSTM, SVM, TFT
  * Mohammadali: XGBoost, RF, DT
  * Aristotelis: CNN
  * Hristiyan: LSTM (assisting Szymon)
* All members must log model tasks on Trello and keep it updated.

---

#### Agenda item 3: Dashboard Progress & Demo

##### Key Discussions

* Aristotelis showed a working dashboard with three pages: overview, ticker details, and prediction logs.
* Team suggested simplifying focus to one main page: ticker details + alert logs.
* Team emphasized that heatmaps and top movers are irrelevant to current scope.
* Everyone agreed the details page with prediction gauge and logs is the strongest part.
* A color palette suggestion site was shared to improve design coherence.

##### Decisions Made

* Aristotelis will refine dashboard based on feedback and focus on relevant event prediction data only.
* Alert log section will be improved with textual summaries following visuals.
* Deliverables must be pushed to GitHub and linked in Trello with PR links.

##### Unresolved Issues

* Overview page's usefulness remains questionable due to data limitations.

---

#### Agenda item 4: Resampling Strategy and Dataset Versions

##### Key Discussions

* Szymon discussed resampling of time-series data and shared graphs showing yearly closing price.
* Team discussed lack of experience with resampling but confirmed visual results looked valid.
* Mohammadali confirmed he was still working with Apple stock but would now try BKNG data.

##### Decisions Made

* Use resampled data where applicable and compare results to original.
* Szymon to upload resampling notebook to GitHub for team use.

---

#### Agenda item 5: Task planning for next 24h

##### Key Discussions

* Mohammadali plans to finish three models by the end of the day.
* Discussion on what to work on next included model comparison preparation and result ranking.
* Mohammadali will create a results table showing accuracy and performance of each model.

##### Decisions Made

* Model ranking table will be used to decide final models for dashboard integration.
* Next standup will include model selection based on performance summaries.

---

### Action Items

| Task                                                    | Responsible | Deadline   |
| ------------------------------------------------------- | ----------- | ---------- |
| Finalize EDA summary and push to GitHub                 | Szymon      | 06-06-2025 |
| Finish XGB, RF, DT models + performance table           | Mohammadali | 05-06-2025 |
| Implement CNN-based time series classification model    | Aristotelis | 06-06-2025 |
| Refactor dashboard based on team feedback               | Aristotelis | 05-06-2025 |
| Push resampling notebook to GitHub                      | Szymon      | 04-06-2025 |
| Log all model/dash tasks and progress updates on Trello | All         | Ongoing    |

> *Make sure all tasks and deadlines are also logged in Trello or your task manager.*

---

### Next Meeting

**Date**: `05/06/2025`
**Time**: `13:00 - 13:30`
**Location**: `Teams Call`
**Chair**: `Szymon Chirowski (242621)`
**Minutes Taker**: `Szymon Chirowski (242621)`
