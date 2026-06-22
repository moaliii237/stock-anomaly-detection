# Agenda and Minutes *27/05/2025*

## Meeting Agenda - *Team 13* - *27/05/2025*

**Group Number**: 13  
**Date**: `27/05/2025`  
**Time**: `09:58 - 10:40`  
**Location**: `Teams Meeting`  
**Chair**: `Szymon Chirowski (242621)`  
**Minutes Taker**: `Szymon Chirowski (242621)`  
**Attendees**:  

- `Szymon Chirowski (242621)`  
- `Hristiyan Georgiev (241226)`  
- `Mohammadali Jaberi (244437)`  
- `Aristotelis Protopapas (234062)`
- `Elavendan Rajendran`

---

### Agenda Items

| # | Topic                                                                 | Presenter             | Duration          | Notes                                                                          |
|---|-----------------------------------------------------------------------|------------------------|-------------------|--------------------------------------------------------------------------------|
| 1 | Retrospective Recording & Attendance                                  |      Elavendan            | 5 mins            | Discussed recording requirements and importance of attendance.                 |
| 2 | News Event Data Review                                                | Elavendan, Hristiyan      | 10 mins           | Issues with data formatting, reliability (broken links), and structure.        |
| 3 | Project Idea Clarification (Event Prediction vs. Outlier Removal)     | All                   | 5 mins            | Confirmed primary goal is event prediction, news sentiment is optional.        |
| 4 | Input/Output Granularity & Event Definition                           | All                   | 5 mins            | Minute-level data, predicting 5 event types in a 30-min window.                |
| 5 | Project Timeline & Deliverables                                       | Elavendan                 | 3 mins            | EDA, final table, model building, backtesting, UI, submission deadlines.       |
| 6 | Backtesting Strategy                                                  | Elavendan                 | 3 mins            | Explained approach with cutoff dates.                                          |
| 7 | EDA Review (Hristiyan's work) & Data Validation                       | Hristiyan, Elavendan      | 7 mins            | Reviewed features, identified potential data issues (e.g., large price jumps). |
| 8 | Meeting Minutes Template                                              | Szymon, Hristiyan     | 2 mins            | Agreed to use Markdown template instead of Word.                               |
| 9 | Task Distribution & Trello Documentation                              | Szymon                | 2 mins            | Discussed roles and need to document in Trello.                                |

---

## Meeting Minutes - *Team 13* - *27/05/2025*

**Group Number**: 13  
**Date**: `27/05/2025`  
**Time**: `09:58 - 10:40`  
**Location**: `Teams Meeting`  
**Chair**: `Szymon Chirowski (242621)`  
**Minutes Taker**: `Szymon Chirowski (242621)`  
**Attendees**:  

- `Szymon Chirowski (242621)` - on time  
- `Hristiyan Georgiev (241226)` - on time  
- `Mohammadali Jaberi (244437)` - on time  
- `Aristotelis Protopapas (234062)` - on time  
- `Elavendan Rajendran` - on time

---

### Discussion Summary

#### Agenda item 1: Retrospective Recording & Attendance

##### Key Discussions - agenda item 1

- Need to record sessions for evidencing.
- Mohammadali and Aristotelis warned about missing retrospectives and impact on grades.

##### Decisions Made - agenda item 1

- Future retrospectives must be attended by all.
- Recordings or documentation are needed for evidencing.

---

#### Agenda item 2: News Event Data Review

##### Key Discussions - agenda item 2

- Data is unstructured, broken, and links may be unreliable.
- Sentiment polarity scores might not be used.
- Hugging Face model proposed for sentiment.

##### Decisions Made - - agenda item 2

- News event data is optional and not ready for use.
- If used, a reliable source is needed.

##### Unresolved Issues - agenda item 2

- Data source reliability and formatting.

---

#### Agenda item 3: Project Idea Clarification (Event Prediction vs. Outlier Removal)

##### Key Discussions - agenda item 3

- Confusion on predicting events vs. removing outliers.
- Primary goal: event prediction from last retrospective.
- News sentiment is optional.

##### Decisions Made - agenda item 3

- Project will focus on predicting specific market events.
- News data is optional.

---

#### Agenda item 4: Input/Output Granularity & Event Definition

##### Key Discussions - agenda item 4

- Predicting 5 event types.
- Window: next 30 minutes using minute-level data.

##### Decisions Made - agenda item 4

- Predict specific event types within a 30-minute window.
- Use defined event categories.

---

#### Agenda item 5: Project Timeline & Deliverables

##### Key Discussions - agenda item 5

- EDA by this week.
- Final table by next Tuesday.
- Model building week 5.
- Backtesting & UI week 6.
- Submission week 7.

##### Decisions Made - agenda item 5

- Timeline agreed upon.

---

#### Agenda item 6: Backtesting Strategy

##### Key Discussions - agenda item 6

- Use cutoff dates (e.g., 6/18 months ago) for performance evaluation.
- Minimum of three backtests.

##### Decisions Made - agenda item 6

- Implement multiple cutoff date strategy.

##### Unresolved Issues - agenda item 6

- Visualization of classification backtest results.

---

#### Agenda item 7: EDA Review (Hristiyan's work) & Data Validation

##### Key Discussions - agenda item 7

- Engineered features presented.
- 2% events vs. 98% normal.
- Apple stock data has a large price jump.

##### Decisions Made - agenda item 7

- Hristiyan's EDA to be validated by another team member.
- Validation to be documented in Trello.

##### Unresolved Issues - agenda item 7

- Cause of Apple stock anomaly.

---

#### Agenda item 8: Meeting Minutes Template

##### Key Discussions - agenda item 8

- Word template formatting and time consumption.
- Markdown template proposed.

##### Decisions Made - agenda item 8

- Markdown template approved.
- Submit as PDF, one file per meeting.

---

#### Agenda item 9: Task Distribution & Trello Documentation

##### Key Discussions - agenda item 9

- Szymon shared task plan:
  - Hristiyan: Data Engineering
  - Szymon: Team Lead, Modeling
  - Mohammadali: Supporting Models, XAI
  - Aristotelis: Dashboard/UI

##### Decisions Made - agenda item 9

- Task split agreed.
- Trello documentation with clear deliverables planned.

##### Unresolved Issues - agenda item 9

- Detailed task breakdown and progress tracking in Trello.

---

### Action Items

| Task                                                 | Responsible                     | Deadline         |
|------------------------------------------------------|----------------------------------|------------------|
| Complete EDA                                         | Hristiyan                        | End of this week |
| Prepare final data table                             | Hristiyan                        | Next Tuesday     |
| Validate Hristiyan's EDA and data                    | Szymon (offered)                 | Next Tuesday     |
| Document EDA validation in Trello                    | Validator                        | Next Tuesday     |
| Investigate Apple stock data discrepancy             | Hristiyan                        | Next Tuesday     |
| Document task distribution and plan in Trello        | Szymon & team                    | ASAP             |
| Figure out evidencing for missed retrospectives      | Team                             | ASAP             |
| Start model building                                 | Szymon, Mohammadali, Hristiyan   | After EDA/Table  |
| Adopt Markdown template for future meeting minutes   | All (Szymon to possibly lead)    | Next meeting     |

---

### Next Meeting

**Date**: `28-05-2025`  
**Time**: `17:00`
**Location**: `Teams Call`  
**Chair**: `Szymon Chirowski (242621)`  
**Minutes Taker**: `Szymon Chirowski (242621)`  
