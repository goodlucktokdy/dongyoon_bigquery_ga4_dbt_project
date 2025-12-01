{{ config(
    materialized='table' 
) }}

SELECT
  -- 1. 시간 정보
  event_date,
  TIMESTAMP_MICROS(event_timestamp) AS event_timestamp,

  -- 2. 사용자 식별
  user_pseudo_id,
  user_id AS member_id,
  COALESCE(user_id, user_pseudo_id) AS master_id,
  IF(user_id IS NOT NULL, 1, 0) AS is_member,

  -- 3. 트래픽 소스 (세션 기준)
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'source') AS session_source,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'medium') AS session_medium,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'campaign') AS session_campaign,

  -- 4. 세션 및 참여 정보
  CONCAT(user_pseudo_id, '-', (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id')) AS session_unique_id,
  (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id,
  COALESCE((SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'session_engaged'), 0) AS is_engaged,
  COALESCE((SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'engagement_time_msec'), 0) AS engagement_time_msec,

  -- 5. 페이지 및 프로모션 정보
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_title') AS page_title,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS page_url,
  
  -- [핵심] 프로모션 배너 이름 (items 안에 숨어있음)
  COALESCE(item.promotion_name, '(not set)') AS promotion_name,

  -- [분석용] 행동 타입 정의 (View Item & Promotion 중심)
  CASE
      WHEN event_name = 'view_item' THEN 'Product Detail'
      WHEN event_name = 'add_to_cart' THEN 'Cart Action'
      WHEN event_name = 'begin_checkout' THEN 'Checkout Start'
      WHEN event_name = 'purchase' THEN 'Purchase'
      WHEN event_name = 'view_promotion' THEN 'Promotion View'
      WHEN event_name = 'select_promotion' THEN 'Promotion Click'
      ELSE 'Browsing'
  END AS action_type,

  -- 6. 이벤트 및 매출
  event_name,
  ecommerce.transaction_id,
  ecommerce.purchase_revenue,

  -- 7. 상품 정보
  item.item_id,
  item.item_name,
  item.item_category,
  item.price AS item_price,
  item.quantity AS item_quantity,
  (IFNULL(item.price, 0) * IFNULL(item.quantity, 0)) AS item_revenue_calc,
  -- 8. 환경 정보 
  device.category AS device_category,  -- mobile, desktop, tablet
  geo.country
FROM
  -- sources.yml에서 정의한 이름을 불러옵니다 (매우 중요!)
  {{ source('ga4', 'events') }}
  LEFT JOIN UNNEST(items) AS item
WHERE
  -- 2020년 12월 데이터만 추출
  _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'