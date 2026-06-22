## Team and my contribution

A 6-person first-year team project. The core LSTM model, the Flask dashboard and the PostgreSQL data warehouse were written by my teammates, mainly Szymon Chirowski and Hristiyan Georgiev. My contribution was the data exploration and documentation, not the production code. I publish it for completeness, with full credit to the team.

---

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/mZzpfrRm)

# Team 13 - NASDAQ-100 Data Challenge (Block D, Year 1)

This is the group project for Team 13. We worked with stock market data for the
client Move Tickers and built a two stage LSTM model to detect market events,
plus a small dashboard to show the results. The full team write up is in the
`Deliverables` folder (reports, meeting minutes, the model code, and the data
warehouse).

The team was Szymon, Hristiyan, Aristotelis and me (Mohammadali Jaberi, 244437).
Most of the model code and the dashboard were written by my teammates. I do not
want to take credit for their code.

My part in this project was the data side and the documentation. I did the early
data exploration (EDA) to understand the NASDAQ-100 data before we picked
features, and I helped with the data collection and pre-processing. I also wrote
documentation, including the data warehouse README and notes, and I kept meeting
minutes. So if you look at the code files, most of them are not mine, but the
EDA and the docs are where I spent my time.

How to run: the main project lives in `Deliverables/project`. It uses Poetry, so
`poetry install` then `poetry run` the scripts there (see that folder's README).
The data warehouse loader is in `Deliverables/ILO6/data-warehouse`. Database
details are read from environment variables, copy `.env.example` to `.env` and
fill in your own values first.

Result: the two stage model splits the problem into "is there an event" and then
"which event", which worked better for us than one single classifier on this
imbalanced data. Details and numbers are in the final report under
`Deliverables/ILO3`.

The rest of this file is the original project brief from the school.

## Project - Uncovering Market Intelligence: A NASDAQ-100 Data Challenge

In this block, you will be working with __Move Tickers__, an innovative SaaS company that specialises in delivering real-time financial market updates and interactive data visualisation tools. They are expanding into the wealth management space, focusing on developing advanced trading strategies, including high-frequency trading and algorithmic models. Their goal is to leverage data and machine learning to uncover hidden patterns in the stock market and gain a competitive edge.

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
