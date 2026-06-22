## Team and my contribution

A 6-person first-year team project. The core LSTM model, the Flask dashboard and the PostgreSQL data warehouse were written by my teammates, mainly Szymon Chirowski and Hristiyan Georgiev. My contribution was the data exploration and documentation, not the production code. I publish it for completeness, with full credit to the team.

---

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/mZzpfrRm)

# Team 13 - NASDAQ-100 Market Event Detection (Block D, Year 1)

This is the group project for Team 13. The client was Move Tickers, a fictional
SaaS company that wanted an early warning tool for short-term market events. We
built a two stage LSTM that looks at minute-by-minute price data for one ticker,
BKNG (Booking Holdings), and tries to flag crashes, dips and rallies as they
happen. Around the model there is a small Flask dashboard that replays test data
and shows the predictions live, and a PostgreSQL star-schema warehouse for the
engineered data. The full team write up is in the `Deliverables` folder
(reports, meeting minutes, the model code, the warehouse and the MLOps notes).

The team was Szymon, Hristiyan, Aristotelis and me (Mohammadali Jaberi, 244437).
Most of the model code and the dashboard were written by my teammates. I do not
want to take credit for their code.

## What I actually did

My part was the data side and the documentation, not the production code.

- I wrote the exploratory data analysis notebook in
  `Deliverables/extra/eda/eda.ipynb`. This was the work I put the most into. I
  loaded the engineered BKNG minute dataset (378,649 rows, 56 columns, from
  2018-02-27 to the end of 2024) and went through it before we committed to any
  features. I cut the data off at 2024-12-31 because too much was missing after
  that, kept only regular trading hours (09:30 to 16:00), checked for missing
  values, and measured the gaps in the timeline (215 missing business days out
  of 1,785, about 12 percent). I plotted per-year close prices, the last 10,000
  minutes of price and volume and 30-minute realized volatility, RSI-14 with the
  70/30 lines, distributions of the main features, a correlation heatmap to spot
  features that move together, hourly volume and volatility patterns, box plots
  of features split by event type, and a mutual information ranking to see which
  features carried signal for the event label. That ranking and the event-type
  comparisons fed into the feature decisions the team made.
- I helped with data collection and the early pre-processing.
- I wrote documentation, including the data warehouse README in
  `Deliverables/ILO6/data-warehouse/README.md`, and I kept meeting minutes (in
  `Deliverables/ILO1`). I am also listed as one of the package authors in
  `pyproject.toml`.

If you open the code, most of it is not mine. The EDA notebook and the docs are
where I spent my time, and that is the honest split.

## What the system does

The problem is anomaly detection on a very imbalanced time series. Most minutes
are "normal", and the events we care about are rare. The team's idea was to not
ask one model to do everything, so they split it into two stages:

- Stage 1 is a binary classifier: is the next 30 minutes a normal period or an
  event. The label column is `event_in_30min` and the model reads a sliding
  window of 30 minutes of features.
- Stage 2 only runs on the windows Stage 1 flagged as events, and classifies the
  type: dip, rally or crash.

The event labels come from a rule on the 30-minute forward price move: a drop of
3 percent or more is a crash, a 1 to 3 percent drop is a dip, a rise of 3 percent
or more is a rally, and anything inside about plus or minus 1 percent is normal.

The dashboard in `Deliverables/project/src/project/app/app.py` is a Flask app
that does not retrain anything. It loads the saved models and a test CSV, then
replays the test points one at a time through `/api/next_point`, lazy-loading the
model on the first call, and draws the price line with the predicted events on
top. The historical chart background is the last 50,000 minutes before the test
window.

## How the model is built

This is my teammates' work (mainly Szymon and Hristiyan), but here is how it
works, since the EDA fed into it.

- Both stages start with a `Conv1D` plus `MaxPooling1D` block to pick up local
  shape in the 30-minute window, then a bidirectional LSTM. Stage 1 is small
  (16 units). Stage 2 is deeper, with two stacked bidirectional LSTM layers and
  two dense blocks, because telling crash from dip from rally is harder than
  telling event from normal.
- Imbalance is handled in two places. Stage 1 uses balanced class weights from
  scikit-learn. Stage 2 uses SMOTE oversampling on the flattened sequences for
  the rare event classes.
- Training uses the AdamW optimizer, dropout and L2 regularization, batch
  normalization, and callbacks for early stopping, learning-rate reduction on
  plateau, model checkpointing, and a custom overfitting monitor. The trained
  models and the preprocessing objects are saved as `stage1_lstm.keras`,
  `stage2_lstm.keras` and `preprocessing_artifacts.pkl`.

The data warehouse (in `Deliverables/ILO6/data-warehouse`) is a PostgreSQL star
schema: a `fact_stock_metrics_minute` table with dimension tables for time,
technical indicators, market conditions and price ratios.

## Tech stack

Python 3.11, Poetry, TensorFlow / Keras, scikit-learn, imbalanced-learn (SMOTE),
pandas, NumPy, matplotlib and seaborn (the EDA plots), Flask for the dashboard,
SQLAlchemy and psycopg2 with PostgreSQL for the warehouse, pytest with
pytest-cov for tests, and Sphinx for the API docs. Code style is black and
flake8.

## Results, honestly

The two stage idea is sound for imbalanced data, but the model we shipped did
not perform well, and I would rather show the real numbers than pretend.

From the saved training run in `Deliverables/project/src/project/run_lstm.ipynb`:

- The binary split is very imbalanced. In training there were 256,203 normal
  windows against 42,807 events; in the held-out test set 34,773 normal against
  2,577 events.
- Stage 1 reached about 72 percent training accuracy on the binary task.
- Stage 2, classifying dip / rally / crash, only reached about 39 percent.
- End to end on the final test set the whole system scored about 27 percent
  overall accuracy, a crash detection rate (recall) around 6 to 11 percent, and
  a false alarm rate near 90 percent.

So the pipeline runs and the engineering around it works, but it is not a useful
predictor yet. The hard parts are the extreme class imbalance, how rare and
sudden the real events are at minute resolution, and that a 30-minute window may
just not carry enough warning. If I came back to it I would start from the
feature side, since that is the part I know, and rethink the labels and the
window length before adding more model capacity.

## How to run

The main project lives in `Deliverables/project` and uses Poetry.

```bash
cd Deliverables/project
poetry install
poetry run python src/project/app/app.py
```

Tests and docs:

```bash
poetry run pytest          # the suite has around 186 tests
cd docs && make html       # Sphinx HTML docs
```

The app does not generate data or models on startup. It needs the engineered
CSVs (`BKNG_engineering.csv`, `BKNG_engineering_test.csv`) and the three saved
model artifacts to be present first. Those files are not committed here. The
CSVs come from the pre-processing step and the artifacts come from training
(`run_lstm.ipynb`) or from `create_artifacts.py`. The original minute data was
shared by the university over OneDrive and is not redistributed.

For the warehouse loader in `Deliverables/ILO6/data-warehouse` you need a running
PostgreSQL instance. Database details are read from environment variables, so
copy `.env.example` to `.env`, fill in your own values, create the schema with
`warehouse-creation.sql`, then run `load_data_to_db.py`.

## Repo layout

- `Deliverables/extra/eda/eda.ipynb` - my EDA notebook.
- `Deliverables/project` - the model, the Flask dashboard, the tests and the docs.
- `Deliverables/ILO6/data-warehouse` - PostgreSQL schema and load scripts.
- `Deliverables/ILO3`, `ILO4` - the written report and proposal.
- `Deliverables/ILO9/mlops` - the MLOps notebook (Poetry, linting, logging, docs).
- `Deliverables/ILO1` - agendas and meeting minutes.

The rest of this file is the original project brief from the school.

## Project - Uncovering Market Intelligence: A NASDAQ-100 Data Challenge

In this block, you will be working with __Move Tickers__, an innovative SaaS company that specialises in delivering real-time financial market updates and interactive data visualisation tools. They are expanding into the wealth management space, focusing on developing advanced trading strategies, including high-frequency trading and algorithmic models. Their goal is to use data and machine learning to uncover hidden patterns in the stock market and gain a competitive edge.

As data scientists collaborating with __Move Tickers__, your role is to help them analyse financial data, build predictive models, and develop trading strategies that are both effective and data-driven. You will explore time series forecasting, classification, clustering, and anomaly detection, and evaluate the performance of different strategies through backtesting. Additionally, __Move Tickers__ is looking for practical solutions, such as dashboards for market monitoring or alert systems that notify users when key events happen.

Throughout this project, you are encouraged to ask critical questions, propose new insights, and validate your findings with solid analysis. Whether you focus on stock price prediction, market correlations, or strategy optimisation, your work should aim to provide clear recommendations and actionable insights that __Move Tickers__ can integrate into their platform. And remember, quality and accuracy matter more than quantity!

Before you get started, take some time to visit the [Move Tickers company page](https://adsai.buas.nl/Study%20Content/Stock%20Market/Client.html). There, you'll find additional background information, their current initiatives, and the specific questions they are looking to answer. Understanding their mission and priorities will help you align your analysis with their real-world needs and deliver solutions that add real value.

For your project in this block, you will have access to a rich set of financial data already prepared for you. You are encouraged to select the data that is most relevant to your project goals. You may also enhance your analysis by adding external datasets if you believe they provide additional value. If you plan to do so, make sure to first discuss it with your mentor. You can focus on a single company or explore multiple ones, depending on the scope of your analysis and the strategy you wish to develop.

The datasets provided cover a wide range of financial and economic indicators, including:

- **NASDAQ-100 Minute Data:** Minute-by-minute stock prices over the last 10 years, ideal for short-term or high-frequency trading analysis.
- **NASDAQ-100 Daily Data:** Daily stock prices over the last 10 years, useful for mid- and long-term trend analysis.
- **NASDAQ-100 Index Data:** Tracks the overall index performance and broader market trends.
- **Top 10 Global Stock Indices:** Provides insights into international market movements.
- **Exchange Rates and Macroeconomic Indicators:** Key factors such as GDP growth, inflation, interest rates, and market volatility (VIX) to help contextualise financial trends.
- **Commodity Prices:** Crude oil and gold prices to assess their impact on market behaviour.

To better understand the details of each dataset, its structure, and its potential applications, please visit the  [Data Overview Page](https://adsai.buas.nl/Study%20Content/Stock%20Market/1.%20Project%20Data%20Overview%20and%20Guidance.html). Familiarising yourself with the available data will help you make informed decisions about which datasets to use and how to apply them effectively in your project.
