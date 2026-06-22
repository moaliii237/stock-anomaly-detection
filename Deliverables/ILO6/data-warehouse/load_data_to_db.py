import logging
import os
from io import StringIO

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockDataWarehouseLoader:
    def __init__(self, db_config):
        """
        Initialize the data warehouse loader

        db_config should be a dictionary with:
        {
            'host': 'localhost',
            'database': 'group_13_warehouse',
            'user': 'your_username',
            'password': 'your_password',
            'port': '5432'
        }
        """
        self.db_config = db_config
        self.connection = None

    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = False
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def clear_existing_data(self, symbol='BKNG'):
        """Clear existing data for the given symbol"""
        try:
            cursor = self.connection.cursor()

            # Clear fact table first (due to foreign key constraints)
            cursor.execute("DELETE FROM fact_stock_metrics_minute WHERE symbol = %s", (symbol,))
            fact_count = cursor.rowcount

            # Clear dimension tables
            cursor.execute("DELETE FROM dim_technical_indicators_minute WHERE symbol = %s", (symbol,))
            tech_count = cursor.rowcount

            cursor.execute("DELETE FROM dim_market_conditions_minute WHERE symbol = %s", (symbol,))
            market_count = cursor.rowcount

            cursor.execute("DELETE FROM dim_price_ratios_minute WHERE symbol = %s", (symbol,))
            ratio_count = cursor.rowcount

            # Clear time dimension (only if no other symbols use it)
            cursor.execute("""
                           DELETE
                           FROM dim_time_minute
                           WHERE time_key NOT IN (SELECT DISTINCT time_key
                                                  FROM fact_stock_metrics_minute)
                           """)
            time_count = cursor.rowcount

            self.connection.commit()

            logger.info(f"Cleared existing data for {symbol}:")
            logger.info(f"  - Fact records: {fact_count}")
            logger.info(f"  - Technical indicators: {tech_count}")
            logger.info(f"  - Market conditions: {market_count}")
            logger.info(f"  - Price ratios: {ratio_count}")
            logger.info(f"  - Time dimension: {time_count}")

        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error clearing existing data: {e}")
            raise
        finally:
            cursor.close()

    def generate_time_key(self, trading_datetime):
        """Generate a unique time key from datetime"""
        # Convert to timestamp and multiply by 1000000 to avoid collisions
        return int(trading_datetime.timestamp() * 1000000)

    def safe_int_convert(self, value, default=0):
        """Safely convert value to integer, handling NaN and float values"""
        if pd.isna(value) or value is None:
            return default
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default

    def safe_float_convert(self, value, default=None):
        """Safely convert value to float, handling NaN values"""
        if pd.isna(value) or value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def safe_bool_convert(self, value, default=False):
        """Safely convert value to boolean"""
        if pd.isna(value) or value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'y']
        try:
            return bool(int(float(value)))
        except (ValueError, TypeError):
            return default

    def calculate_trigonometric_features(self, df):
        """Calculate trigonometric features for ML if not present in CSV"""
        # Check if these columns already exist in the CSV
        trig_columns = ['Adv_Hour_sin', 'Adv_Hour_cos', 'Adv_DayOfWeek_sin',
                        'Adv_DayOfWeek_cos', 'Adv_Month_sin', 'Adv_Month_cos']

        for col in trig_columns:
            if col not in df.columns:
                if col == 'Adv_Hour_sin':
                    df['adv_hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                elif col == 'Adv_Hour_cos':
                    df['adv_hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                elif col == 'Adv_DayOfWeek_sin':
                    # Convert 1-7 to 0-6 for calculation, then back
                    day_normalized = (df['day_of_week'] - 1) / 7
                    df['adv_dayofweek_sin'] = np.sin(2 * np.pi * day_normalized)
                elif col == 'Adv_DayOfWeek_cos':
                    day_normalized = (df['day_of_week'] - 1) / 7
                    df['adv_dayofweek_cos'] = np.cos(2 * np.pi * day_normalized)
                elif col == 'Adv_Month_sin':
                    month_normalized = (df['month_of_year'] - 1) / 12
                    df['adv_month_sin'] = np.sin(2 * np.pi * month_normalized)
                elif col == 'Adv_Month_cos':
                    month_normalized = (df['month_of_year'] - 1) / 12
                    df['adv_month_cos'] = np.cos(2 * np.pi * month_normalized)
            else:
                # Use existing columns, just rename for consistency
                if col == 'Adv_Hour_sin':
                    df['adv_hour_sin'] = df[col]
                elif col == 'Adv_Hour_cos':
                    df['adv_hour_cos'] = df[col]
                elif col == 'Adv_DayOfWeek_sin':
                    df['adv_dayofweek_sin'] = df[col]
                elif col == 'Adv_DayOfWeek_cos':
                    df['adv_dayofweek_cos'] = df[col]
                elif col == 'Adv_Month_sin':
                    df['adv_month_sin'] = df[col]
                elif col == 'Adv_Month_cos':
                    df['adv_month_cos'] = df[col]

        return df

    def prepare_time_dimension(self, df):
        """Prepare data for dim_time_minute table"""
        logger.info("Preparing time dimension data")

        # Parse datetime and extract components
        df['trading_datetime'] = pd.to_datetime(df['Date'])
        df['trading_date'] = df['trading_datetime'].dt.date
        df['year'] = df['trading_datetime'].dt.year
        df['month'] = df['trading_datetime'].dt.month
        df['day'] = df['trading_datetime'].dt.day
        df['hour'] = df['trading_datetime'].dt.hour
        df['minute'] = df['trading_datetime'].dt.minute

        # Use existing day_of_week and month_of_year from CSV if available
        if 'day_of_week' not in df.columns:
            # If not in CSV, calculate and convert to 1-7 range
            df['day_of_week'] = df['trading_datetime'].dt.dayofweek + 1  # Convert 0-6 to 1-7

        if 'month_of_year' not in df.columns:
            df['month_of_year'] = df['trading_datetime'].dt.month

        # Ensure day_of_week is in correct range (1-7) and is integer
        df['day_of_week'] = df['day_of_week'].apply(lambda x: self.safe_int_convert(x, 1))
        if df['day_of_week'].min() == 0:
            df['day_of_week'] = df['day_of_week'] + 1

        # Ensure month_of_year is integer
        df['month_of_year'] = df['month_of_year'].apply(lambda x: self.safe_int_convert(x, 1))

        # Calculate minutes since open if hours_since_open exists
        if 'hours_since_open' in df.columns:
            df['minutes_since_open'] = df['hours_since_open'].apply(
                lambda x: self.safe_int_convert(self.safe_float_convert(x, 0) * 60, 0)
            )
        else:
            df['minutes_since_open'] = 0

        # Calculate trigonometric features
        df = self.calculate_trigonometric_features(df)

        # Generate time key
        df['time_key'] = df['trading_datetime'].apply(self.generate_time_key)

        # Map additional columns from CSV if they exist
        column_mapping = {
            'Adv_Minutes_to_Market_Close': 'minutes_to_market_close',
            'Adv_Is_Market_Open_Period': 'is_market_open_period',
            'Adv_Is_Market_Close_Period': 'is_market_close_period'
        }

        for csv_col, db_col in column_mapping.items():
            if csv_col in df.columns:
                if 'is_market' in db_col:
                    df[db_col] = df[csv_col].apply(lambda x: self.safe_bool_convert(x, False))
                else:
                    df[db_col] = df[csv_col].apply(lambda x: self.safe_int_convert(x))
            else:
                if 'is_market' in db_col:
                    df[db_col] = False
                else:
                    df[db_col] = None

        # Select columns for time dimension
        time_cols = [
            'time_key', 'trading_datetime', 'trading_date', 'year', 'month', 'day',
            'day_of_week', 'month_of_year', 'hour', 'minute', 'hours_since_open',
            'minutes_since_open', 'minutes_to_market_close', 'is_market_open_period',
            'is_market_close_period', 'adv_hour_sin', 'adv_hour_cos', 'adv_dayofweek_sin',
            'adv_dayofweek_cos', 'adv_month_sin', 'adv_month_cos'
        ]

        # Select only available columns
        available_cols = [col for col in time_cols if col in df.columns]

        return df[available_cols].drop_duplicates(subset=['time_key'])

    def prepare_technical_indicators(self, df, symbol='BKNG'):
        """Prepare data for dim_technical_indicators_minute table"""
        logger.info("Preparing technical indicators data")

        tech_indicators = df[['trading_datetime']].copy()
        tech_indicators['symbol'] = symbol

        # Define min and max values for NUMERIC(8,6) fields
        min_value_8_6 = -99.999999
        max_value_8_6 = 99.999999

        # Map technical indicator columns with precision constraints
        indicator_mapping = {
            # Higher precision columns (NUMERIC(10,4)) - no capping needed
            'MA_50': ('ma_50', None),
            'MA_100D_proxy': ('ma_100d_proxy', None),
            'MA_200': ('ma_200', None),
            'BB_MA_20': ('bb_ma_20', None),
            'BB_upper': ('bb_upper', None),
            'BB_lower': ('bb_lower', None),
            'tenkan_sen': ('tenkan_sen', None),
            'kijun_sen': ('kijun_sen', None),
            'senkou_span_a': ('senkou_span_a', None),
            'senkou_span_b': ('senkou_span_b', None),
            'chikou_span': ('chikou_span', None),

            # NUMERIC(8,4) - different constraint for RSI
            'RSI_14': ('rsi_14', (0, 100)),  # RSI has its own constraint

            # NUMERIC(8,6) columns - need capping
            'MA_50_slope': ('ma_50_slope', (min_value_8_6, max_value_8_6)),
            'MA_200_slope': ('ma_200_slope', (min_value_8_6, max_value_8_6)),
            'MACD': ('macd', (min_value_8_6, max_value_8_6)),
            'MACD_Signal': ('macd_signal', (min_value_8_6, max_value_8_6)),
            'MACD_Hist': ('macd_hist', (min_value_8_6, max_value_8_6)),
            'ROC_5': ('roc_5', (min_value_8_6, max_value_8_6)),
            'ROC_15': ('roc_15', (min_value_8_6, max_value_8_6)),
            'ATR_14': ('atr_14', (min_value_8_6, max_value_8_6)),
            'realized_vol_5min': ('realized_vol_5min', (min_value_8_6, max_value_8_6)),
            'realized_vol_30min': ('realized_vol_30min', (min_value_8_6, max_value_8_6)),
            'vol_of_vol_ATR14_30min': ('vol_of_vol_atr14_30min', (min_value_8_6, max_value_8_6)),
            'Adv_VoV_ATR14_10min': ('vol_of_vol_atr14_10min', (min_value_8_6, max_value_8_6)),
            'BB_std_20': ('bb_std_20', (min_value_8_6, max_value_8_6)),
            'BB_width': ('bb_width', (min_value_8_6, max_value_8_6))
        }

        for csv_col, (db_col, constraint) in indicator_mapping.items():
            if csv_col in df.columns:
                values = df[csv_col].apply(lambda x: self.safe_float_convert(x))

                # Apply constraints if specified
                if constraint:
                    min_val, max_val = constraint
                    values = values.clip(lower=min_val, upper=max_val)

                tech_indicators[db_col] = values
            else:
                tech_indicators[db_col] = None

        return tech_indicators

    def prepare_market_conditions(self, df, symbol='BKNG'):
        """Prepare data for dim_market_conditions_minute table"""
        logger.info("Preparing market conditions data")

        market_conditions = df[['trading_datetime']].copy()
        market_conditions['symbol'] = symbol

        # Map market condition columns with proper type conversion
        condition_mapping = {
            '52w_high': ('ref_52w_high', 'float'),
            '52w_low': ('ref_52w_low', 'float'),
            'Volume_30D_avg': ('ref_volume_30d_avg', 'float'),
            'Close/52w_high': ('close_52w_high_ratio', 'float'),
            'Close/52w_low': ('close_52w_low_ratio', 'float'),
            'days_since_ATH': ('days_since_ath', 'int'),
            'covid_period': ('covid_period', 'bool'),
            'high_vol_regime_90q': ('high_vol_regime_90q', 'bool'),
            'bull_market_proxy': ('bull_market_proxy', 'bool'),
            'bear_market_proxy': ('bear_market_proxy', 'bool'),
            'oversold_RSI_30': ('oversold_rsi_30', 'bool'),
            'liquidity_dryup_signal': ('liquidity_dryup_signal', 'bool'),
            'positive_divergence_placeholder': ('positive_divergence_placeholder', 'bool'),
            'high_volume_bar_60min_95p': ('high_volume_bar_60min_95p', 'bool'),
            'consec_high_volume_bars_5': ('consec_high_volume_bars_5', 'int'),
            'event': ('event', 'str')
        }

        for csv_col, (db_col, dtype) in condition_mapping.items():
            if csv_col in df.columns:
                if dtype == 'int':
                    market_conditions[db_col] = df[csv_col].apply(lambda x: self.safe_int_convert(x, 0))
                elif dtype == 'float':
                    market_conditions[db_col] = df[csv_col].apply(lambda x: self.safe_float_convert(x))
                elif dtype == 'bool':
                    market_conditions[db_col] = df[csv_col].apply(lambda x: self.safe_bool_convert(x, False))
                else:  # string
                    market_conditions[db_col] = df[csv_col].fillna('normal')
            else:
                # Set defaults for missing columns
                if dtype == 'bool':
                    market_conditions[db_col] = False
                elif dtype == 'int':
                    market_conditions[db_col] = 0
                elif dtype == 'str':
                    market_conditions[db_col] = 'normal'
                else:
                    market_conditions[db_col] = None

        return market_conditions

    def prepare_price_ratios(self, df, symbol='BKNG'):
        """Prepare data for dim_price_ratios_minute table"""
        logger.info("Preparing price ratios data")

        price_ratios = df[['trading_datetime']].copy()
        price_ratios['symbol'] = symbol

        # Map price ratio columns
        ratio_mapping = {
            'Close/MA_50': 'close_ma_50_ratio',
            'Close/MA_200': 'close_ma_200_ratio',
            'Close/VWAP': 'close_vwap_ratio',
            'Price_30D_zscore': 'price_30d_zscore',
            'Volume/30D_avg': 'volume_30d_avg_ratio',
            'volume_acceleration': 'volume_acceleration',
            'Adv_ATR_Norm_by_Price': 'atr_norm_by_price',
            'Adv_RSI_x_VolChange': 'rsi_x_volchange'
        }

        # Define min and max values for NUMERIC(8,6)
        min_value = -99.999999
        max_value = 99.999999

        for csv_col, db_col in ratio_mapping.items():
            if csv_col in df.columns:
                # Convert and cap values
                values = df[csv_col].apply(lambda x: self.safe_float_convert(x))
                # Cap extreme values to fit NUMERIC(8,6) constraints
                values = values.clip(lower=min_value, upper=max_value)
                price_ratios[db_col] = values
            else:
                price_ratios[db_col] = None

        return price_ratios

    def prepare_fact_table(self, df, symbol='BKNG'):
        """Prepare data for fact_stock_metrics_minute table"""
        logger.info("Preparing fact table data")

        fact_data = df[['trading_datetime']].copy()
        fact_data['symbol'] = symbol
        fact_data['time_key'] = df['time_key']

        # Map fact table columns with proper type conversion
        fact_mapping = {
            'Open': ('open_price', 'float'),
            'High': ('high_price', 'float'),
            'Low': ('low_price', 'float'),
            'Close': ('close_price', 'float'),
            'Volume': ('volume', 'int'),
            'Transactions': ('transactions', 'int'),
            'VWAP': ('vwap', 'float'),
            'Adv_Dollar_Volume': ('dollar_volume', 'float'),
            'intraday_amplitude': ('intraday_amplitude', 'float'),
            'overnight_gap': ('overnight_gap', 'float'),
            'price_acceleration': ('price_acceleration', 'float'),
            'Volume_pct_change_1min': ('volume_pct_change_1min', 'float'),
            'volume_rolling_avg_short': ('volume_rolling_avg_short', 'float'),
            'volume_rolling_avg_long': ('volume_rolling_avg_long', 'float'),
            'Volume_intraday_zscore': ('volume_intraday_zscore', 'float')
        }

        for csv_col, (db_col, dtype) in fact_mapping.items():
            if csv_col in df.columns:
                if dtype == 'int':
                    fact_data[db_col] = df[csv_col].apply(lambda x: self.safe_int_convert(x, 0))
                else:  # float
                    fact_data[db_col] = df[csv_col].apply(lambda x: self.safe_float_convert(x))
            else:
                fact_data[db_col] = None

        return fact_data

    def bulk_insert_data_with_conflict_handling(self, table_name, dataframe, columns, conflict_columns=None):
        """Efficiently insert data using PostgreSQL COPY command with conflict handling"""
        logger.info(f"Bulk inserting {len(dataframe)} rows into {table_name}")

        try:
            cursor = self.connection.cursor()

            if conflict_columns:
                # Use INSERT ... ON CONFLICT for tables with potential duplicates
                placeholders = ', '.join(['%s'] * len(columns))
                conflict_cols = ', '.join(conflict_columns)

                insert_query = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT ({conflict_cols}) DO NOTHING
                """

                # Prepare data for batch insert
                data_tuples = []
                for _, row in dataframe.iterrows():
                    row_data = []
                    for col in columns:
                        value = row[col]
                        if pd.isna(value):
                            row_data.append(None)
                        else:
                            row_data.append(value)
                    data_tuples.append(tuple(row_data))

                psycopg2.extras.execute_batch(cursor, insert_query, data_tuples, page_size=1000)

            else:
                # Use COPY command for better performance when no conflicts expected
                df_copy = dataframe[columns].copy()

                # Handle NaN values by replacing with None
                for col in df_copy.columns:
                    df_copy[col] = df_copy[col].where(pd.notnull(df_copy[col]), None)

                csv_buffer = StringIO()
                df_copy.to_csv(csv_buffer, index=False, header=False, na_rep='\\N')
                csv_buffer.seek(0)

                cursor.copy_from(
                    csv_buffer,
                    table_name,
                    sep=',',
                    columns=columns,
                    null='\\N'
                )

            self.connection.commit()
            logger.info(f"Successfully inserted data into {table_name}")

        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error inserting data into {table_name}: {e}")
            raise
        finally:
            cursor.close()

    def get_market_condition_keys(self, market_conditions_df):
        """Get market condition keys after insertion"""
        cursor = self.connection.cursor()

        # Create a temporary table for efficient lookup
        temp_query = """
                     SELECT market_condition_key, trading_datetime, symbol
                     FROM dim_market_conditions_minute
                     WHERE (trading_datetime, symbol) IN %s \
                     """

        datetime_symbol_pairs = list(zip(
            market_conditions_df['trading_datetime'],
            market_conditions_df['symbol']
        ))

        cursor.execute(temp_query, (tuple(datetime_symbol_pairs),))
        results = cursor.fetchall()
        cursor.close()

        # Create mapping dictionary
        key_mapping = {(dt, sym): key for key, dt, sym in results}

        return market_conditions_df.apply(
            lambda row: key_mapping.get((row['trading_datetime'], row['symbol'])),
            axis=1
        )

    def load_csv_to_warehouse(self, csv_file_path, symbol='BKNG', chunk_size=10000, clear_existing=True):
        """Main method to load CSV data into the warehouse"""
        logger.info(f"Starting data load from {csv_file_path}")

        try:
            # Clear existing data if requested
            if clear_existing:
                logger.info(f"Clearing existing data for symbol: {symbol}")
                self.clear_existing_data(symbol)

            # Read CSV file
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loaded {len(df)} rows from CSV")

            # Process data in chunks for memory efficiency
            for chunk_start in range(0, len(df), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(df))
                chunk_df = df.iloc[chunk_start:chunk_end].copy()

                logger.info(f"Processing chunk {chunk_start // chunk_size + 1}: rows {chunk_start} to {chunk_end}")

                # Prepare dimension data
                time_dim = self.prepare_time_dimension(chunk_df)
                tech_indicators = self.prepare_technical_indicators(chunk_df, symbol)
                market_conditions = self.prepare_market_conditions(chunk_df, symbol)
                price_ratios = self.prepare_price_ratios(chunk_df, symbol)

                # Insert dimension data with conflict handling
                if not time_dim.empty:
                    time_columns = [col for col in [
                        'time_key', 'trading_datetime', 'trading_date', 'year', 'month', 'day',
                        'day_of_week', 'month_of_year', 'hour', 'minute', 'hours_since_open',
                        'minutes_since_open', 'minutes_to_market_close', 'is_market_open_period',
                        'is_market_close_period', 'adv_hour_sin', 'adv_hour_cos', 'adv_dayofweek_sin',
                        'adv_dayofweek_cos', 'adv_month_sin', 'adv_month_cos'
                    ] if col in time_dim.columns]

                    self.bulk_insert_data_with_conflict_handling(
                        'dim_time_minute',
                        time_dim,
                        time_columns,
                        conflict_columns=['time_key']
                    )

                # Insert technical indicators
                tech_columns = ['trading_datetime', 'symbol'] + [col for col in tech_indicators.columns
                                                                 if col not in ['trading_datetime', 'symbol']]
                self.bulk_insert_data_with_conflict_handling(
                    'dim_technical_indicators_minute',
                    tech_indicators,
                    tech_columns,
                    conflict_columns=['trading_datetime', 'symbol']
                )

                # Insert market conditions
                market_columns = ['trading_datetime', 'symbol'] + [col for col in market_conditions.columns
                                                                   if col not in ['trading_datetime', 'symbol']]
                self.bulk_insert_data_with_conflict_handling(
                    'dim_market_conditions_minute',
                    market_conditions,
                    market_columns,
                    conflict_columns=['trading_datetime', 'symbol']
                )

                # Insert price ratios
                ratio_columns = ['trading_datetime', 'symbol'] + [col for col in price_ratios.columns
                                                                  if col not in ['trading_datetime', 'symbol']]
                self.bulk_insert_data_with_conflict_handling(
                    'dim_price_ratios_minute',
                    price_ratios,
                    ratio_columns,
                    conflict_columns=['trading_datetime', 'symbol']
                )

                # Get market condition keys for fact table
                market_condition_keys = self.get_market_condition_keys(market_conditions)

                # Prepare and insert fact data
                chunk_df['time_key'] = time_dim['time_key'].values
                fact_data = self.prepare_fact_table(chunk_df, symbol)
                fact_data['market_condition_key'] = market_condition_keys.values

                fact_columns = ['trading_datetime', 'symbol', 'time_key', 'market_condition_key'] + \
                               [col for col in fact_data.columns
                                if col not in ['trading_datetime', 'symbol', 'time_key', 'market_condition_key']]
                self.bulk_insert_data_with_conflict_handling(
                    'fact_stock_metrics_minute',
                    fact_data,
                    fact_columns,
                    conflict_columns=['trading_datetime', 'symbol']
                )

            logger.info("Data load completed successfully")

        except Exception as e:
            logger.error(f"Error during data load: {e}")
            raise


def main():
    # Database configuration is read from environment variables so no
    # credentials are stored in the code. See .env.example for the names.
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'group_13_warehouse'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', '5432')
    }

    # Initialize loader
    loader = StockDataWarehouseLoader(db_config)

    try:
        # Connect to database
        loader.connect()

        # Load data from CSV
        csv_file_path = 'data/BKNG_engineering.csv'  # Replace with your CSV file path
        loader.load_csv_to_warehouse(
            csv_file_path,
            symbol='BKNG',
            chunk_size=5000,
            clear_existing=True  # Set to False if you want to keep existing data
        )

    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        # Disconnect from database
        loader.disconnect()


if __name__ == "__main__":
    main()
