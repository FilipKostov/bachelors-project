SELECT item_name,
       AVG(points_per_day) AS avg_points_per_day
FROM (
  SELECT item_name, date_trunc('day', "timestamp") AS d, COUNT(*) AS points_per_day
  FROM market_prices
  GROUP BY item_name, date_trunc('day', "timestamp")
) t
GROUP BY item_name
HAVING AVG(points_per_day) > 3
ORDER BY avg_points_per_day DESC
LIMIT 50;