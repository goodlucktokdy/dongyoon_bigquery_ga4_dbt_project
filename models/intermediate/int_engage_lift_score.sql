{{ config(materialized='table') }}

WITH session_scores AS (
    SELECT
        session_unique_id,
        user_pseudo_id,
        
        -- Lift 기반 점수 산정
        SUM(CASE 
            WHEN event_name = 'view_item' THEN 5              -- Lift 4.6 -> 5점
            WHEN event_name = 'view_search_results' THEN 3    -- Lift 2.9 -> 3점
            WHEN event_name = 'add_to_cart' THEN 12           -- Lift 11.8 -> 12점
            WHEN event_name = 'begin_checkout' THEN 31        -- Lift 30.6 -> 31점
            WHEN event_name = 'add_payment_info' THEN 47      -- Lift 46.5 -> 47점
            
            -- 그 외 단순 방문 
            ELSE 1
        END) AS engagement_score
    FROM {{ ref('stg_events') }}
    GROUP BY 1, 2
),

ranked AS (
    SELECT
        *,
        -- 점수 줄세우기 (백분위 계산)
        PERCENT_RANK() OVER (ORDER BY engagement_score DESC) as pct_rank
    FROM session_scores
)

SELECT
    session_unique_id,
    user_pseudo_id,
    engagement_score,
    -- 등급 부여 (상위 20% / 50% / 나머지)
    CASE 
        WHEN pct_rank <= 0.2 THEN 'High Intent'   -- 상위 20% (진성 유저)
        WHEN pct_rank <= 0.5 THEN 'Medium Intent' -- 상위 20~50% (탐색 유저)
        ELSE 'Low Intent'                         -- 하위 50% (이탈 유저)
    END AS engagement_grade
FROM ranked