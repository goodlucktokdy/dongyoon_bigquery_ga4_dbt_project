{{ config(materialized='table') }}

SELECT
    session_hour,
    COUNT(*) AS sessions,
    SUM(has_purchase) AS purchased,
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS cvr
FROM {{ ref('int_session_funnel') }}
GROUP BY session_hour
ORDER BY session_hour