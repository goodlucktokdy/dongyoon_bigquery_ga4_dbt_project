{{ config(materialized='table') }}

SELECT
    IFNULL(session_source, '(direct)') AS source,
    IFNULL(session_medium, '(none)') AS medium,
    COUNT(*) AS sessions,
    SUM(has_purchase) AS purchased,
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS cvr
FROM {{ ref('int_session_funnel') }}
GROUP BY 1, 2
HAVING COUNT(*) >= 50
ORDER BY sessions DESC
LIMIT 20