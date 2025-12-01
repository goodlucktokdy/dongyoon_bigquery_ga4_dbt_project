{{ config(materialized='table') }}

WITH purchase_items AS (
    SELECT
        transaction_id,
        session_unique_id,
        item_name
    FROM {{ ref('stg_events') }}
    WHERE event_name = 'purchase' 
      AND transaction_id IS NOT NULL
)
SELECT
    a.session_unique_id,
    a.transaction_id,
    a.item_name AS product_A,
    b.item_name AS product_B
FROM purchase_items a
JOIN purchase_items b
  ON a.transaction_id = b.transaction_id
 AND a.item_name < b.item_name -- 중복 제거 (A-B만 남김)