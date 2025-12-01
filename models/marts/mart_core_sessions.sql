{{ config(materialized='table') }}

WITH scores AS (
    SELECT * FROM {{ ref('int_engage_lift_score') }}
),

paths AS (
    SELECT * FROM {{ ref('int_session_paths') }}
)

SELECT
    s.session_unique_id,
    s.user_pseudo_id,
    
    -- 1. 세션 등급 및 점수
    s.engagement_grade,     -- High / Medium 
    s.engagement_score,     -- Lift 기반 점수

    -- 2. 이동 경로 (event_name 기반)
    p.full_path,
    p.path_length,
    COALESCE(p.is_converted, 0) AS is_converted, -- 구매여부 (1 or 0)

    -- 3. '놓친 기회(Missed Opportunity)' 플래그
    -- 구매 안 하고(Converted=0) 나간 경우
    CASE 
        WHEN COALESCE(p.is_converted, 0) = 0 THEN TRUE 
        ELSE FALSE 
    END AS is_missed_opportunity

FROM scores s
LEFT JOIN paths p ON s.session_unique_id = p.session_unique_id

-- 의미 없는 단순 세션 제거
WHERE 
    -- 조건 
    s.engagement_grade IN ('High Intent','Medium Intent')