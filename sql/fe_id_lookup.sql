-- ============================================================
-- FE_ID DISCOVERY SCRIPT
-- ============================================================
-- Purpose: find the FE_IDs for the bridge_condition_model features
-- that are NOT yet wired into bridge_data_extract.sql.
--
-- We don't know your AssetWise install's metadata table name for sure,
-- so Step 1 finds it, then Step 2 searches it by keyword.
-- Run each step, inspect the results, adjust table/column names if the
-- guesses below are wrong, and re-run.
-- ============================================================

-- ------------------------------------------------------------
-- STEP 1: Find candidate metadata table(s) that describe FE_IDs
-- (name/description of each feature/element).
-- ------------------------------------------------------------
SELECT TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_NAME LIKE '%FEATURE%'
   OR TABLE_NAME LIKE '%ELEMENT%'
ORDER BY TABLE_NAME;

-- Once you find the right table (likely something like TBLFEATURES),
-- inspect its columns:
-- SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
-- WHERE TABLE_NAME = 'TBLFEATURES';

-- ------------------------------------------------------------
-- STEP 2: Search that table by keyword for each missing field.
-- Replace TBLFEATURES / FE_ID / FE_NAME below with the real
-- table + column names once confirmed in Step 1.
-- ------------------------------------------------------------
SELECT FE_ID, FE_NAME
FROM TBLFEATURES
WHERE
       FE_NAME LIKE '%structure length%'
    OR FE_NAME LIKE '%span length%'
    OR FE_NAME LIKE '%max span%'
    OR FE_NAME LIKE '%deck width%'
    OR FE_NAME LIKE '%roadway width%'
    OR FE_NAME LIKE '%curb to curb%'
    OR FE_NAME LIKE '%average daily traffic%'
    OR FE_NAME LIKE '%adt%'
    OR FE_NAME LIKE '%year built%'
    OR FE_NAME LIKE '%year reconstruct%'
    OR FE_NAME LIKE '%skew%'
    OR FE_NAME LIKE '%number of spans%'
    OR FE_NAME LIKE '%spans in main%'
    OR FE_NAME LIKE '%spans in approach%'
    OR FE_NAME LIKE '%owner%'
    OR FE_NAME LIKE '%maintenance respons%'
    OR FE_NAME LIKE '%structure kind%'
    OR FE_NAME LIKE '%structure type%'
    OR FE_NAME LIKE '%deck type%'
    OR FE_NAME LIKE '%wearing surface%'
    OR FE_NAME LIKE '%membrane%'
    OR FE_NAME LIKE '%deck protection%'
    OR FE_NAME LIKE '%design load%'
    OR FE_NAME LIKE '%approach roadway width%'
    OR FE_NAME LIKE '%functional class%'
ORDER BY FE_NAME;

-- ------------------------------------------------------------
-- STEP 2b (FOLLOW-UP, 2026-07-10): only what's still unresolved
-- after the first pass. Narrower on purpose so the result set is
-- small enough to paste back in full without getting cut off.
-- ------------------------------------------------------------
SELECT FE_ID, FE_NAME, FE_DESCRIPTION
FROM TBLFEATURES
WHERE
       FE_NAME LIKE '%traffic%'
    OR FE_NAME LIKE '%adt%'
    OR FE_NAME LIKE '%reconstruct%'
    OR FE_NAME LIKE '%rehab%'
    OR FE_NAME LIKE '%widen%'
    OR FE_NAME LIKE '%work event%'
    OR FE_NAME LIKE '%membrane%'
    OR FE_NAME LIKE '%functional class%'
    OR FE_NAME LIKE '%route%'
ORDER BY FE_NAME;

-- ------------------------------------------------------------
-- STEP 2c (FOLLOW-UP): what does the FE_ID range used for BH17_1..BH17_100
-- in the original inventory query (2306021, 2306052, 2306083, ... 2309090)
-- actually represent? Needed before we can decide whether num_spans_main/
-- num_spans_approach or any structure/deck/material fields live in here.
-- ------------------------------------------------------------
SELECT FE_ID, FE_NAME, FE_DESCRIPTION
FROM TBLFEATURES
WHERE FE_ID IN (2306021, 2306052, 2306083, 2306114, 2306145)
ORDER BY FE_ID;

-- ------------------------------------------------------------
-- STEP 2d (FOLLOW-UP): confirm the B.SP (span material/type) instance-1
-- FE_IDs for structure_kind/structure_type/deck_type/wearing_surface/
-- deck_protection, per the dictionary rows already seen:
--   2303005 B.SP.04 Span Material (1)   -> structure_kind?
--   2303007 B.SP.06 Span Type           -> structure_type?
--   2303010 B.SP.09 Deck Material and Type (1) -> deck_type?
--   2303011 B.SP.10 Wearing Surface (1) -> wearing_surface?
--   2303012 B.SP.11 Deck Protective System (1) -> deck_protection?
-- Pull a sample to sanity-check the values look like categories, not codes/dates:
SELECT TOP 20 AS_ID, FE_ID, CV_VALUE
FROM TBLCURRENTVALUES
WHERE FE_ID IN (2303005, 2303007, 2303010, 2303011, 2303012)
ORDER BY AS_ID, FE_ID;

-- ------------------------------------------------------------
-- STEP 3 (optional sanity check): once you have a candidate FE_ID,
-- confirm it actually has data and looks like the right shape/values.
-- ------------------------------------------------------------
-- SELECT TOP 20 AS_ID, CV_VALUE
-- FROM TBLCURRENTVALUES
-- WHERE FE_ID = <candidate_fe_id>;
