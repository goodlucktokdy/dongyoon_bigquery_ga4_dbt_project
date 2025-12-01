{{ config(materialized='table') }}

SELECT
    session_unique_id,
    -- 행동 순서를 문자열로 연결 (예: Product Detail > Cart Action > Checkout)
    STRING_AGG(event_name, ' > ' ORDER BY event_timestamp ASC) AS full_path,
    -- 경로 길이 (몇 단계나 거쳤는지)
    COUNT(*) AS path_length,
    -- 구매 여부 (전환 확인)
    MAX(CASE WHEN action_type = 'Purchase' THEN 1 ELSE 0 END) AS is_converted
FROM {{ ref('stg_events') }}
GROUP BY 1