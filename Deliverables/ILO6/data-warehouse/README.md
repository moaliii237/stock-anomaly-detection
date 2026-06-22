# Data Warehouse

This folder holds the PostgreSQL star-schema warehouse for the BKNG minute data
and the scripts to create and load it.

## Files

- `warehouse-creation.sql` - creates the fact and dimension tables.
- `load_data_to_db.py` - reads an engineered CSV and loads it into the warehouse.
- `get_sample_data.py` - takes the first N rows of a CSV for quick testing.

## Requirements

- Python 3.11 (>= 3.11, < 3.12)
- Poetry for the dependencies (see `Deliverables/project/pyproject.toml`,
  which includes `psycopg2`, `pandas`, and `numpy`).
- A running PostgreSQL database with the schema from `warehouse-creation.sql`.
- The engineered CSV (for example `data/BKNG_engineering.csv`) produced by the
  data preprocessing step.

## Database credentials

The loader reads connection details from environment variables, so no
credentials are stored in the code. Copy `.env.example` in the repo root to
`.env` and set your own values, then export them before running:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=group_13_warehouse
export DB_USER=your_username
export DB_PASSWORD=your_password
```

## Run

1. Create the schema:

   ```bash
   psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f warehouse-creation.sql
   ```

2. Load a CSV (edit `csv_file_path` in `main()` if your file is elsewhere):

   ```bash
   python load_data_to_db.py
   ```
