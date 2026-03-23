CREATE TABLE market_prices (
    id SERIAL PRIMARY KEY,
    item_name TEXT NOT NULL,
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    median_price NUMERIC,
    volume INTEGER,
    currency TEXT,
    appid INTEGER,
    retrieved_at TIMESTAMP WITHOUT TIME ZONE,
    UNIQUE(item_name, timestamp)
);