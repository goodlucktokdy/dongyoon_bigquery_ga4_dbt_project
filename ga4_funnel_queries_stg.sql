-- ============================================================
-- GA4 이커머스 퍼널 분석 쿼리
-- 기준 테이블: stg_events (이미 정제된 staging 테이블)
-- 기간: 2020년 12월 1일 ~ 31일
-- ============================================================


-- ============================================================
-- 1. 전체 퍼널 집계 (세션 기준)
-- ============================================================
WITH session_funnel AS (
    SELECT
        session_unique_id,
        MAX(CASE WHEN event_name = 'session_start' THEN 1 ELSE 0 END) AS has_session_start,
        MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END) AS has_view_item,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS has_add_to_cart,
        MAX(CASE WHEN event_name = 'begin_checkout' THEN 1 ELSE 0 END) AS has_begin_checkout,
        MAX(CASE WHEN event_name = 'add_payment_info' THEN 1 ELSE 0 END) AS has_add_payment_info,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS has_purchase
    FROM {{ ref('stg_events') }}
    GROUP BY session_unique_id
)

SELECT
    COUNT(*) AS total_sessions,
    SUM(has_view_item) AS step1_view_item,
    SUM(has_add_to_cart) AS step2_add_to_cart,
    SUM(has_begin_checkout) AS step3_begin_checkout,
    SUM(has_add_payment_info) AS step4_add_payment_info,
    SUM(has_purchase) AS step5_purchase,
    
    -- 전환율 (전체 세션 대비)
    ROUND(SUM(has_view_item) / COUNT(*) * 100, 2) AS pct_view,
    ROUND(SUM(has_add_to_cart) / COUNT(*) * 100, 2) AS pct_cart,
    ROUND(SUM(has_begin_checkout) / COUNT(*) * 100, 2) AS pct_checkout,
    ROUND(SUM(has_add_payment_info) / COUNT(*) * 100, 2) AS pct_payment,
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS pct_purchase
FROM session_funnel;


-- ============================================================
-- 2. 단계별 이탈률 분석 (mart_funnel_dropoff)
-- ============================================================
WITH session_funnel AS (
    SELECT
        session_unique_id,
        MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END) AS has_view,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS has_cart,
        MAX(CASE WHEN event_name = 'begin_checkout' THEN 1 ELSE 0 END) AS has_checkout,
        MAX(CASE WHEN event_name = 'add_payment_info' THEN 1 ELSE 0 END) AS has_payment,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS has_purchase
    FROM {{ ref('stg_events') }}
    GROUP BY session_unique_id
),

funnel_counts AS (
    SELECT
        COUNT(*) AS total_sessions,
        SUM(has_view) AS viewed,
        SUM(has_cart) AS carted,
        SUM(has_checkout) AS checkout,
        SUM(has_payment) AS payment,
        SUM(has_purchase) AS purchased
    FROM session_funnel
)

SELECT 1 AS step_order, 'Session → View Item' AS step, total_sessions AS from_count, viewed AS to_count,
       total_sessions - viewed AS dropped, ROUND((total_sessions - viewed) / total_sessions * 100, 2) AS drop_rate
FROM funnel_counts
UNION ALL
SELECT 2, 'View Item → Add to Cart', viewed, carted, viewed - carted, ROUND((viewed - carted) / NULLIF(viewed, 0) * 100, 2) FROM funnel_counts
UNION ALL
SELECT 3, 'Add to Cart → Begin Checkout', carted, checkout, carted - checkout, ROUND((carted - checkout) / NULLIF(carted, 0) * 100, 2) FROM funnel_counts
UNION ALL
SELECT 4, 'Begin Checkout → Add Payment', checkout, payment, checkout - payment, ROUND((checkout - payment) / NULLIF(checkout, 0) * 100, 2) FROM funnel_counts
UNION ALL
SELECT 5, 'Add Payment → Purchase', payment, purchased, payment - purchased, ROUND((payment - purchased) / NULLIF(payment, 0) * 100, 2) FROM funnel_counts
ORDER BY step_order;


-- ============================================================
-- 3. 디바이스별 퍼널 분석 (mart_funnel_by_device)
-- ============================================================
WITH session_funnel AS (
    SELECT
        session_unique_id,
        MAX(device_category) AS device_category,
        MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END) AS has_view,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS has_cart,
        MAX(CASE WHEN event_name = 'begin_checkout' THEN 1 ELSE 0 END) AS has_checkout,
        MAX(CASE WHEN event_name = 'add_payment_info' THEN 1 ELSE 0 END) AS has_payment,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS has_purchase
    FROM {{ ref('stg_events') }}
    GROUP BY session_unique_id
)

SELECT
    device_category,
    COUNT(*) AS sessions,
    SUM(has_view) AS viewed,
    SUM(has_cart) AS carted,
    SUM(has_checkout) AS checkout,
    SUM(has_payment) AS payment,
    SUM(has_purchase) AS purchased,
    
    -- 전체 전환율
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS overall_cvr,
    
    -- 단계별 전환율
    ROUND(SUM(has_cart) / NULLIF(SUM(has_view), 0) * 100, 2) AS view_to_cart,
    ROUND(SUM(has_checkout) / NULLIF(SUM(has_cart), 0) * 100, 2) AS cart_to_checkout,
    ROUND(SUM(has_purchase) / NULLIF(SUM(has_checkout), 0) * 100, 2) AS checkout_to_purchase
FROM session_funnel
GROUP BY device_category
ORDER BY sessions DESC;


-- ============================================================
-- 4. 요일별 퍼널 분석 (mart_funnel_by_day)
-- ============================================================
WITH session_funnel AS (
    SELECT
        session_unique_id,
        -- 세션의 첫 이벤트 시간 기준 요일
        MIN(EXTRACT(DAYOFWEEK FROM event_timestamp)) AS session_day,
        MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END) AS has_view,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS has_cart,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS has_purchase
    FROM {{ ref('stg_events') }}
    GROUP BY session_unique_id
)

SELECT
    session_day,
    CASE session_day
        WHEN 1 THEN '일요일'
        WHEN 2 THEN '월요일'
        WHEN 3 THEN '화요일'
        WHEN 4 THEN '수요일'
        WHEN 5 THEN '목요일'
        WHEN 6 THEN '금요일'
        WHEN 7 THEN '토요일'
    END AS day_name,
    COUNT(*) AS sessions,
    SUM(has_view) AS viewed,
    SUM(has_cart) AS carted,
    SUM(has_purchase) AS purchased,
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS cvr
FROM session_funnel
GROUP BY session_day
ORDER BY session_day;


-- ============================================================
-- 5. 시간대별 퍼널 분석 (mart_funnel_by_hour)
-- ============================================================
WITH session_funnel AS (
    SELECT
        session_unique_id,
        MIN(EXTRACT(HOUR FROM event_timestamp)) AS session_hour,
        MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END) AS has_view,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS has_cart,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS has_purchase
    FROM {{ ref('stg_events') }}
    GROUP BY session_unique_id
)

SELECT
    session_hour,
    COUNT(*) AS sessions,
    SUM(has_view) AS viewed,
    SUM(has_cart) AS carted,
    SUM(has_purchase) AS purchased,
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS cvr
FROM session_funnel
GROUP BY session_hour
ORDER BY session_hour;


-- ============================================================
-- 6. 트래픽 소스별 퍼널 분석 (mart_funnel_by_source)
-- ============================================================
WITH session_funnel AS (
    SELECT
        session_unique_id,
        MAX(session_source) AS source,
        MAX(session_medium) AS medium,
        MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END) AS has_view,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS has_cart,
        MAX(CASE WHEN event_name = 'begin_checkout' THEN 1 ELSE 0 END) AS has_checkout,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS has_purchase
    FROM {{ ref('stg_events') }}
    GROUP BY session_unique_id
)

SELECT
    IFNULL(source, '(direct)') AS source,
    IFNULL(medium, '(none)') AS medium,
    COUNT(*) AS sessions,
    SUM(has_view) AS viewed,
    SUM(has_cart) AS carted,
    SUM(has_checkout) AS checkout,
    SUM(has_purchase) AS purchased,
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS cvr,
    ROUND(SUM(has_cart) / NULLIF(SUM(has_view), 0) * 100, 2) AS view_to_cart
FROM session_funnel
GROUP BY source, medium
HAVING COUNT(*) >= 50
ORDER BY sessions DESC
LIMIT 20;


-- ============================================================
-- 7. Sankey용 경로 데이터 (mart_funnel_sankey)
-- ============================================================
WITH session_funnel AS (
    SELECT
        session_unique_id,
        MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END) AS has_view,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS has_cart,
        MAX(CASE WHEN event_name = 'begin_checkout' THEN 1 ELSE 0 END) AS has_checkout,
        MAX(CASE WHEN event_name = 'add_payment_info' THEN 1 ELSE 0 END) AS has_payment,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS has_purchase
    FROM {{ ref('stg_events') }}
    GROUP BY session_unique_id
)

SELECT
    CASE 
        WHEN has_purchase = 1 THEN '5_Purchased'
        WHEN has_payment = 1 THEN '4_Dropped_Payment'
        WHEN has_checkout = 1 THEN '3_Dropped_Checkout'
        WHEN has_cart = 1 THEN '2_Dropped_Cart'
        WHEN has_view = 1 THEN '1_Dropped_View'
        ELSE '0_No_View'
    END AS final_stage,
    COUNT(*) AS session_count
FROM session_funnel
GROUP BY 1
ORDER BY 1;


-- ============================================================
-- 8. 대시보드용 퍼널 마트 테이블 (mart_funnel.csv)
-- ============================================================
{{ config(materialized='table') }}

WITH session_funnel AS (
    SELECT
        session_unique_id,
        MAX(CASE WHEN event_name = 'session_start' THEN 1 ELSE 0 END) AS step0,
        MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END) AS step1,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS step2,
        MAX(CASE WHEN event_name = 'begin_checkout' THEN 1 ELSE 0 END) AS step3,
        MAX(CASE WHEN event_name = 'add_payment_info' THEN 1 ELSE 0 END) AS step4,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS step5
    FROM {{ ref('stg_events') }}
    GROUP BY session_unique_id
),

totals AS (
    SELECT
        COUNT(*) AS sessions,
        SUM(step1) AS viewed,
        SUM(step2) AS carted,
        SUM(step3) AS checkout,
        SUM(step4) AS payment,
        SUM(step5) AS purchased
    FROM session_funnel
)

SELECT 0 AS step_order, 'Total Sessions' AS stage_name, sessions AS session_count, 
       100.0 AS pct_of_total, NULL AS pct_of_previous, NULL AS drop_rate_pct
FROM totals
UNION ALL
SELECT 1, 'View Item', viewed, ROUND(viewed/sessions*100, 2), ROUND(viewed/sessions*100, 2), ROUND((sessions-viewed)/sessions*100, 2) FROM totals
UNION ALL
SELECT 2, 'Add to Cart', carted, ROUND(carted/sessions*100, 2), ROUND(carted/NULLIF(viewed,0)*100, 2), ROUND((viewed-carted)/NULLIF(viewed,0)*100, 2) FROM totals
UNION ALL
SELECT 3, 'Begin Checkout', checkout, ROUND(checkout/sessions*100, 2), ROUND(checkout/NULLIF(carted,0)*100, 2), ROUND((carted-checkout)/NULLIF(carted,0)*100, 2) FROM totals
UNION ALL
SELECT 4, 'Add Payment Info', payment, ROUND(payment/sessions*100, 2), ROUND(payment/NULLIF(checkout,0)*100, 2), ROUND((checkout-payment)/NULLIF(checkout,0)*100, 2) FROM totals
UNION ALL
SELECT 5, 'Purchase', purchased, ROUND(purchased/sessions*100, 2), ROUND(purchased/NULLIF(payment,0)*100, 2), ROUND((payment-purchased)/NULLIF(payment,0)*100, 2) FROM totals
ORDER BY step_order;


-- ============================================================
-- 9. 캠페인별 퍼널 분석 (mart_funnel_by_campaign)
-- ============================================================
WITH session_funnel AS (
    SELECT
        session_unique_id,
        MAX(session_campaign) AS campaign,
        MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END) AS has_view,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS has_cart,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS has_purchase,
        MAX(purchase_revenue) AS revenue
    FROM {{ ref('stg_events') }}
    GROUP BY session_unique_id
)

SELECT
    IFNULL(campaign, '(not set)') AS campaign,
    COUNT(*) AS sessions,
    SUM(has_view) AS viewed,
    SUM(has_cart) AS carted,
    SUM(has_purchase) AS purchased,
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS cvr,
    ROUND(SUM(revenue), 2) AS total_revenue,
    ROUND(SUM(revenue) / NULLIF(SUM(has_purchase), 0), 2) AS avg_order_value
FROM session_funnel
GROUP BY campaign
HAVING COUNT(*) >= 30
ORDER BY sessions DESC;


-- ============================================================
-- 10. 회원/비회원 퍼널 비교 (mart_funnel_by_member)
-- ============================================================
WITH session_funnel AS (
    SELECT
        session_unique_id,
        MAX(is_member) AS is_member,
        MAX(CASE WHEN event_name = 'view_item' THEN 1 ELSE 0 END) AS has_view,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS has_cart,
        MAX(CASE WHEN event_name = 'begin_checkout' THEN 1 ELSE 0 END) AS has_checkout,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS has_purchase,
        MAX(purchase_revenue) AS revenue
    FROM {{ ref('stg_events') }}
    GROUP BY session_unique_id
)

SELECT
    CASE WHEN is_member = 1 THEN '회원' ELSE '비회원' END AS member_type,
    COUNT(*) AS sessions,
    SUM(has_view) AS viewed,
    SUM(has_cart) AS carted,
    SUM(has_checkout) AS checkout,
    SUM(has_purchase) AS purchased,
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS cvr,
    ROUND(SUM(has_cart) / NULLIF(SUM(has_view), 0) * 100, 2) AS view_to_cart,
    ROUND(SUM(has_purchase) / NULLIF(SUM(has_cart), 0) * 100, 2) AS cart_to_purchase,
    ROUND(SUM(revenue), 2) AS total_revenue
FROM session_funnel
GROUP BY is_member
ORDER BY sessions DESC;
