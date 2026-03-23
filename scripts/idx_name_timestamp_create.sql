CREATE INDEX IF NOT EXISTS idx_market_prices_item_ts
  ON market_prices (item_name, "timestamp");