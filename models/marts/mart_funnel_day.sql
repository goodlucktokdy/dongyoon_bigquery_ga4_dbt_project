{{ config(materialized='table') }}

SELECT
    session_day_of_week AS session_day,
    CASE session_day_of_week
        WHEN 1 THEN '일요일' WHEN 2 THEN '월요일' WHEN 3 THEN '화요일'
        WHEN 4 THEN '수요일' WHEN 5 THEN '목요일' WHEN 6 THEN '금요일' WHEN 7 THEN '토요일'
    END AS day_name,
    COUNT(*) AS sessions,
    SUM(has_purchase) AS purchased,
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS cvr
FROM {{ ref('int_session_funnel') }}
GROUP BY 1, 2
ORDER BY 1