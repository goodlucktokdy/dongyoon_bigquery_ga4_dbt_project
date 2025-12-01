{{ config(materialized='table') }}

SELECT
    session_unique_id,
    
    -- 1. 세션 속성 정보 (Dimension)
    -- (MAX 함수는 세션 내 하나의 값만 가져오기 위함)
    MAX(session_source) AS session_source,
    MAX(session_medium) AS session_medium,
    MAX(session_campaign) AS session_campaign,
    MAX(device_category) AS device_category,
    MAX(is_member) AS is_member,
    
    -- 2. 시간 정보
    MIN(event_timestamp) AS session_start_at,
    MIN(EXTRACT(HOUR FROM event_timestamp)) AS session_hour,
    MIN(EXTRACT(DAYOFWEEK FROM event_timestamp)) AS session_day_of_week,

    -- 3. 퍼널 도달 여부 (Flags)
    MAX(CASE WHEN event_name = 'session_start' THEN 1 ELSE 0 END) AS has_session_start,
    MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END) AS has_view_item,
    MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS has_add_to_cart,
    MAX(CASE WHEN event_name = 'begin_checkout' THEN 1 ELSE 0 END) AS has_begin_checkout,
    MAX(CASE WHEN event_name = 'add_payment_info' THEN 1 ELSE 0 END) AS has_add_payment_info,
    MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS has_purchase,
    
    -- 4. 매출 정보
    MAX(purchase_revenue) AS revenue

FROM {{ ref('stg_events') }}
GROUP BY session_unique_id