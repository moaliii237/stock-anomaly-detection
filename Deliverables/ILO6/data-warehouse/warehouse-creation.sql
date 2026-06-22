-- Truncate tables before dropping them
TRUNCATE TABLE fact_stock_metrics_minute CASCADE;
TRUNCATE TABLE dim_price_ratios_minute CASCADE;
TRUNCATE TABLE dim_market_conditions_minute CASCADE;
TRUNCATE TABLE dim_technical_indicators_minute CASCADE;
TRUNCATE TABLE dim_time_minute CASCADE;

-- Drop tables if they exist (in reverse dependency order)
DROP TABLE IF EXISTS fact_stock_metrics_minute CASCADE;
DROP TABLE IF EXISTS dim_price_ratios_minute CASCADE;
DROP TABLE IF EXISTS dim_market_conditions_minute CASCADE;
DROP TABLE IF EXISTS dim_technical_indicators_minute CASCADE;
DROP TABLE IF EXISTS dim_time_minute CASCADE;

CREATE TABLE dim_time_minute (
    time_key BIGINT PRIMARY KEY,                                -- NEW: Generated surrogate key
    trading_datetime TIMESTAMP WITHOUT TIME ZONE NOT NULL,      -- SOURCE: Date (parsed as timestamp)
    trading_date DATE NOT NULL,                                 -- SOURCE: Date (date part only)

    -- Date Components
    year INTEGER NOT NULL,                                      -- NEW: Extracted from Date
    month INTEGER NOT NULL,                                     -- NEW: Extracted from Date
    day INTEGER NOT NULL,                                       -- NEW: Extracted from Date
    day_of_week INTEGER NOT NULL,                               -- SOURCE: day_of_week
    month_of_year INTEGER NOT NULL,                             -- SOURCE: month_of_year

    -- Time Components
    hour INTEGER NOT NULL,                                      -- NEW: Extracted from Date
    minute INTEGER NOT NULL,                                    -- NEW: Extracted from Date

    -- Market Session Features
    hours_since_open NUMERIC(4,2),                              -- SOURCE: hours_since_open
    minutes_since_open INTEGER,                                 -- NEW: Calculated from hours_since_open
    minutes_to_market_close INTEGER,                            -- SOURCE: Adv_Minutes_to_Market_Close
    is_market_open_period BOOLEAN DEFAULT FALSE,                -- SOURCE: Adv_Is_Market_Open_Period
    is_market_close_period BOOLEAN DEFAULT FALSE,               -- SOURCE: Adv_Is_Market_Close_Period

    -- Derived time features for ML
    adv_hour_sin NUMERIC(8,6),                                  -- SOURCE: Adv_Hour_sin
    adv_hour_cos NUMERIC(8,6),                                  -- SOURCE: Adv_Hour_cos
    adv_dayofweek_sin NUMERIC(8,6),                             -- SOURCE: Adv_DayOfWeek_sin
    adv_dayofweek_cos NUMERIC(8,6),                             -- SOURCE: Adv_DayOfWeek_cos
    adv_month_sin NUMERIC(8,6),                                 -- SOURCE: Adv_Month_sin
    adv_month_cos NUMERIC(8,6),                                 -- SOURCE: Adv_Month_cos

    -- Constraints
    CONSTRAINT chk_hour CHECK (hour >= 0 AND hour <= 23),
    CONSTRAINT chk_minute CHECK (minute >= 0 AND minute <= 59),
    CONSTRAINT chk_day_of_week CHECK (day_of_week >= 1 AND day_of_week <= 7)
);

-- Indexes for time dimension
CREATE INDEX idx_dim_time_trading_datetime ON dim_time_minute (trading_datetime);
CREATE INDEX idx_dim_time_trading_date ON dim_time_minute (trading_date);
CREATE INDEX idx_dim_time_hour_minute ON dim_time_minute (hour, minute);

CREATE TABLE dim_technical_indicators_minute (
    indicator_key BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,  -- NEW: Generated surrogate key
    trading_datetime TIMESTAMP WITHOUT TIME ZONE NOT NULL,          -- SOURCE: Date (parsed as timestamp)
    symbol VARCHAR(10) NOT NULL,                                    -- NEW: Hardcoded 'BKNG' (or parameterized)

    -- Moving Averages
    ma_50 NUMERIC(10,4),                                            -- SOURCE: MA_50
    ma_100d_proxy NUMERIC(10,4),                                    -- SOURCE: MA_100D_proxy
    ma_200 NUMERIC(10,4),                                           -- SOURCE: MA_200
    ma_50_slope NUMERIC(8,6),                                       -- SOURCE: MA_50_slope
    ma_200_slope NUMERIC(8,6),                                      -- SOURCE: MA_200_slope

    -- Momentum Indicators
    rsi_14 NUMERIC(8,4),                                            -- SOURCE: RSI_14
    macd NUMERIC(8,6),                                              -- SOURCE: MACD
    macd_signal NUMERIC(8,6),                                       -- SOURCE: MACD_Signal
    macd_hist NUMERIC(8,6),                                         -- SOURCE: MACD_Hist
    roc_5 NUMERIC(8,6),                                             -- SOURCE: ROC_5
    roc_15 NUMERIC(8,6),                                            -- SOURCE: ROC_15

    -- Volatility Indicators
    atr_14 NUMERIC(8,6),                                            -- SOURCE: ATR_14
    realized_vol_5min NUMERIC(8,6),                                 -- SOURCE: realized_vol_5min
    realized_vol_30min NUMERIC(8,6),                                -- SOURCE: realized_vol_30min
    vol_of_vol_atr14_30min NUMERIC(8,6),                            -- SOURCE: vol_of_vol_ATR14_30min
    vol_of_vol_atr14_10min NUMERIC(8,6),                            -- SOURCE: Adv_VoV_ATR14_10min

    -- Bollinger Bands
    bb_ma_20 NUMERIC(10,4),                                         -- SOURCE: BB_MA_20
    bb_std_20 NUMERIC(8,6),                                         -- SOURCE: BB_std_20
    bb_upper NUMERIC(10,4),                                         -- SOURCE: BB_upper
    bb_lower NUMERIC(10,4),                                         -- SOURCE: BB_lower
    bb_width NUMERIC(8,6),                                          -- SOURCE: BB_width

    -- Ichimoku Indicators
    tenkan_sen NUMERIC(10,4),                                       -- SOURCE: tenkan_sen
    kijun_sen NUMERIC(10,4),                                        -- SOURCE: kijun_sen
    senkou_span_a NUMERIC(10,4),                                    -- SOURCE: senkou_span_a
    senkou_span_b NUMERIC(10,4),                                    -- SOURCE: senkou_span_b
    chikou_span NUMERIC(10,4),                                      -- SOURCE: chikou_span

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,                 -- NEW: Audit column
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,                 -- NEW: Audit column

    -- Constraints
    CONSTRAINT uk_technical_indicators_datetime_symbol UNIQUE (trading_datetime, symbol),
    CONSTRAINT chk_rsi_range CHECK (rsi_14 >= 0 AND rsi_14 <= 100)
);

-- Indexes for technical indicators
CREATE INDEX idx_tech_indicators_datetime_symbol ON dim_technical_indicators_minute (trading_datetime, symbol);
CREATE INDEX idx_tech_indicators_symbol ON dim_technical_indicators_minute (symbol);
CREATE INDEX idx_tech_indicators_rsi ON dim_technical_indicators_minute (rsi_14);
CREATE INDEX idx_tech_indicators_macd ON dim_technical_indicators_minute (macd);

CREATE TABLE dim_market_conditions_minute (
    market_condition_key INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,   -- NEW: Generated surrogate key
    trading_datetime TIMESTAMP WITHOUT TIME ZONE NOT NULL,                   -- SOURCE: Date (parsed as timestamp)
    symbol VARCHAR(10) NOT NULL,                                             -- NEW: Hardcoded 'BKNG' (or parameterized)

    -- Reference Data
    ref_52w_high NUMERIC(10,4),                                              -- SOURCE: 52w_high
    ref_52w_low NUMERIC(10,4),                                               -- SOURCE: 52w_low
    ref_volume_30d_avg NUMERIC(15,2),                                        -- SOURCE: Volume_30D_avg

    -- Market Context Ratios
    close_52w_high_ratio NUMERIC(8,6),                                       -- SOURCE: Close/52w_high
    close_52w_low_ratio NUMERIC(8,6),                                        -- SOURCE: Close/52w_low
    days_since_ath INTEGER,                                                  -- SOURCE: days_since_ATH

    -- Market Regime Flags
    covid_period BOOLEAN DEFAULT FALSE,                                      -- SOURCE: covid_period
    high_vol_regime_90q BOOLEAN DEFAULT FALSE,                               -- SOURCE: high_vol_regime_90q
    bull_market_proxy BOOLEAN DEFAULT FALSE,                                 -- SOURCE: bull_market_proxy
    bear_market_proxy BOOLEAN DEFAULT FALSE,                                 -- SOURCE: bear_market_proxy

    -- Signal Flags
    oversold_rsi_30 BOOLEAN DEFAULT FALSE,                                   -- SOURCE: oversold_RSI_30
    liquidity_dryup_signal BOOLEAN DEFAULT FALSE,                            -- SOURCE: liquidity_dryup_signal
    positive_divergence_placeholder BOOLEAN DEFAULT FALSE,                   -- SOURCE: positive_divergence_placeholder
    high_volume_bar_60min_95p BOOLEAN DEFAULT FALSE,                         -- SOURCE: high_volume_bar_60min_95p
    consec_high_volume_bars_5 INTEGER DEFAULT 0,                             -- SOURCE: consec_high_volume_bars_5

    -- Event Classification
    event VARCHAR(50) DEFAULT 'normal',                                      -- SOURCE: event

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,                          -- NEW: Audit column

    -- Constraints
    CONSTRAINT uk_market_conditions_datetime_symbol UNIQUE (trading_datetime, symbol),
    CONSTRAINT chk_days_since_ath CHECK (days_since_ath >= 0),
    CONSTRAINT chk_consec_volume_bars CHECK (consec_high_volume_bars_5 >= 0)
);

-- Indexes for market conditions
CREATE INDEX idx_market_conditions_datetime_symbol ON dim_market_conditions_minute (trading_datetime, symbol);
CREATE INDEX idx_market_conditions_regime_flags ON dim_market_conditions_minute (bull_market_proxy, high_vol_regime_90q);
CREATE INDEX idx_market_conditions_signals ON dim_market_conditions_minute (oversold_rsi_30, high_volume_bar_60min_95p);
CREATE INDEX idx_market_conditions_event ON dim_market_conditions_minute (event);

CREATE TABLE dim_price_ratios_minute (
    ratio_key BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,      -- NEW: Generated surrogate key
    trading_datetime TIMESTAMP WITHOUT TIME ZONE NOT NULL,          -- SOURCE: Date (parsed as timestamp)
    symbol VARCHAR(10) NOT NULL,                                    -- NEW: Hardcoded 'BKNG' (or parameterized)

    -- Price Ratios
    close_ma_50_ratio NUMERIC(8,6),                                 -- SOURCE: Close/MA_50
    close_ma_200_ratio NUMERIC(8,6),                                -- SOURCE: Close/MA_200
    close_vwap_ratio NUMERIC(8,6),                                  -- SOURCE: Close/VWAP
    price_30d_zscore NUMERIC(8,6),                                  -- SOURCE: Price_30D_zscore

    -- Volume Ratios
    volume_30d_avg_ratio NUMERIC(8,6),                              -- SOURCE: Volume/30D_avg
    volume_acceleration NUMERIC(8,6),                               -- SOURCE: volume_acceleration

    -- Normalized Metrics
    atr_norm_by_price NUMERIC(8,6),                                 -- SOURCE: Adv_ATR_Norm_by_Price
    rsi_x_volchange NUMERIC(8,6),                                   -- SOURCE: Adv_RSI_x_VolChange

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,                 -- NEW: Audit column

    -- Constraints
    CONSTRAINT uk_price_ratios_datetime_symbol UNIQUE (trading_datetime, symbol)
);

-- Indexes for price ratios
CREATE INDEX idx_price_ratios_datetime_symbol ON dim_price_ratios_minute (trading_datetime, symbol);
CREATE INDEX idx_price_ratios_symbol ON dim_price_ratios_minute (symbol);

-- Create partitioned fact table
CREATE TABLE fact_stock_metrics_minute (
    trading_datetime TIMESTAMP WITHOUT TIME ZONE NOT NULL,      -- SOURCE: Date (parsed as timestamp)
    symbol VARCHAR(10) NOT NULL,                                -- NEW: Hardcoded 'BKNG' (or parameterized)
    time_key BIGINT NOT NULL,                                   -- NEW: Generated foreign key to dim_time_minute
    market_condition_key INTEGER NOT NULL,                      -- NEW: Generated foreign key to dim_market_conditions_minute

    -- Core Price/Volume Metrics
    open_price NUMERIC(10,4),                                   -- SOURCE: Open
    high_price NUMERIC(10,4),                                   -- SOURCE: High
    low_price NUMERIC(10,4),                                    -- SOURCE: Low
    close_price NUMERIC(10,4),                                  -- SOURCE: Close
    volume BIGINT,                                              -- SOURCE: Volume
    transactions INTEGER,                                       -- SOURCE: Transactions
    vwap NUMERIC(10,4),                                         -- SOURCE: VWAP
    dollar_volume NUMERIC(15,2),                                -- SOURCE: Adv_Dollar_Volume

    -- Performance Metrics
    intraday_amplitude NUMERIC(8,6),                            -- SOURCE: intraday_amplitude
    overnight_gap NUMERIC(8,6),                                 -- SOURCE: overnight_gap
    price_acceleration NUMERIC(8,6),                            -- SOURCE: price_acceleration

    -- Volume Analytics
    volume_pct_change_1min NUMERIC(8,6),                        -- SOURCE: Volume_pct_change_1min
    volume_rolling_avg_short NUMERIC(15,2),                     -- SOURCE: volume_rolling_avg_short
    volume_rolling_avg_long NUMERIC(15,2),                      -- SOURCE: volume_rolling_avg_long
    volume_intraday_zscore NUMERIC(8,6),                        -- SOURCE: Volume_intraday_zscore

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,             -- NEW: Audit column

    -- Constraints
    PRIMARY KEY (trading_datetime, symbol),
    CONSTRAINT fk_fact_time_key FOREIGN KEY (time_key) REFERENCES dim_time_minute(time_key),
    CONSTRAINT fk_fact_market_condition_key FOREIGN KEY (market_condition_key) REFERENCES dim_market_conditions_minute(market_condition_key),

    -- Data Quality Checks
    CONSTRAINT chk_prices_positive CHECK (open_price > 0 AND high_price > 0 AND low_price > 0 AND close_price > 0),
    CONSTRAINT chk_high_low CHECK (high_price >= low_price),
    CONSTRAINT chk_volume_positive CHECK (volume >= 0),
    CONSTRAINT chk_transactions_positive CHECK (transactions >= 0)
) PARTITION BY RANGE (trading_datetime);

-- Add foreign keys to fact table
	ALTER TABLE fact_stock_metrics_minute 
	ADD COLUMN technical_indicators_key BIGINT,
	ADD COLUMN price_ratios_key BIGINT;

-- Create foreign key constraints
	ALTER TABLE fact_stock_metrics_minute
	ADD CONSTRAINT fk_fact_technical_indicators 
		FOREIGN KEY (technical_indicators_key) 
		REFERENCES dim_technical_indicators_minute(indicator_key),
	ADD CONSTRAINT fk_fact_price_ratios 
		FOREIGN KEY (price_ratios_key) 
		REFERENCES dim_price_ratios_minute(ratio_key);


-- Create partitions for each year from 2018 to 2024
CREATE TABLE fact_stock_metrics_minute_2018 PARTITION OF fact_stock_metrics_minute
    FOR VALUES FROM ('2018-01-01 00:00:00') TO ('2019-01-01 00:00:00');

CREATE TABLE fact_stock_metrics_minute_2019 PARTITION OF fact_stock_metrics_minute
    FOR VALUES FROM ('2019-01-01 00:00:00') TO ('2020-01-01 00:00:00');

CREATE TABLE fact_stock_metrics_minute_2020 PARTITION OF fact_stock_metrics_minute
    FOR VALUES FROM ('2020-01-01 00:00:00') TO ('2021-01-01 00:00:00');

CREATE TABLE fact_stock_metrics_minute_2021 PARTITION OF fact_stock_metrics_minute
    FOR VALUES FROM ('2021-01-01 00:00:00') TO ('2022-01-01 00:00:00');

CREATE TABLE fact_stock_metrics_minute_2022 PARTITION OF fact_stock_metrics_minute
    FOR VALUES FROM ('2022-01-01 00:00:00') TO ('2023-01-01 00:00:00');

CREATE TABLE fact_stock_metrics_minute_2023 PARTITION OF fact_stock_metrics_minute
    FOR VALUES FROM ('2023-01-01 00:00:00') TO ('2024-01-01 00:00:00');

CREATE TABLE fact_stock_metrics_minute_2024 PARTITION OF fact_stock_metrics_minute
    FOR VALUES FROM ('2024-01-01 00:00:00') TO ('2025-01-01 00:00:00');

-- Indexes for fact table
CREATE INDEX idx_fact_stock_metrics_datetime ON fact_stock_metrics_minute (trading_datetime);
CREATE INDEX idx_fact_stock_metrics_symbol ON fact_stock_metrics_minute (symbol);
CREATE INDEX idx_fact_stock_metrics_volume ON fact_stock_metrics_minute (volume);
CREATE INDEX idx_fact_stock_metrics_close_price ON fact_stock_metrics_minute (close_price);
