-- ============================================================
-- Bridge Condition Model — Data Extract
-- ============================================================
-- Trimmed/aligned version of the AssetWise inventory query, tailored
-- to exactly the columns bridge_condition_model expects (see
-- ../config.yaml). Column aliases below match config.yaml verbatim
-- so the CSV this produces can be dropped straight into
-- data/raw/bridge_data.csv with no renaming step.
--
-- STATUS OF EACH FIELD:
--   [confirmed]            FE_ID carried over from the original working query.
--   [cheatsheet 2026-07-28] FE_ID resolved from the SNBI Transition CheatSheet.
--   [TODO]                 FE_ID still unknown or structurally awkward — see notes.
--
-- IMPORTANT — CURRENT VALUES vs. INSPECTION HISTORY:
--   This query reads TBLCURRENTVALUES, which holds only the LATEST value per
--   bridge → one row per bridge. The training panel (1996-2023, ~20 inspections
--   per bridge) comes from an inspection-history source, NOT this table. Use this
--   query to VERIFY the new SNBI fields are populated. The structural (B.SP.*)
--   fields are static per bridge, so they can be joined here and back-filled across
--   the historical panel by bridge_id; condition ratings and inspection dates must
--   be pulled per-inspection from the history source.
-- ============================================================

-- REPRODUCIBILITY: export this result WITH its header row. src/enrichment.py joins the static
-- physical columns onto the inspection panel by `as_code` (= bridge_id). If a headerless CSV is
-- exported, enrichment.py falls back to EXTRACT_COLUMNS (which must stay in this exact SELECT order).
SELECT
    AST.AS_ID   AS as_id,
    AST.AS_CODE AS as_code,

    -- ---------------- targets [confirmed] ----------------
    -- Order matches SNBI B.C.01-04 / NBI 58,59,60,62: Deck, Superstructure, Substructure, Culvert
    CASE
        WHEN AST.ASSET_STATUS_ID = 2 THEN NULL
        WHEN AST.ASSET_STATUS_ID = 0 THEN COALESCE(BC01.CV_VALUE, BC01_NBI.CV_VALUE)
        ELSE BC01.CV_VALUE
    END AS deck_cond_rating,

    CASE
        WHEN AST.ASSET_STATUS_ID = 2 THEN NULL
        WHEN AST.ASSET_STATUS_ID = 0 THEN COALESCE(BC02.CV_VALUE, BC02_NBI.CV_VALUE)
        ELSE BC02.CV_VALUE
    END AS superstructure_cond_rating,

    CASE
        WHEN AST.ASSET_STATUS_ID = 2 THEN NULL
        WHEN AST.ASSET_STATUS_ID = 0 THEN COALESCE(BC03.CV_VALUE, BC03_NBI.CV_VALUE)
        ELSE BC03.CV_VALUE
    END AS substructure_cond_rating,

    CASE
        WHEN AST.ASSET_STATUS_ID = 2 THEN NULL
        WHEN AST.ASSET_STATUS_ID = 0 THEN COALESCE(BC04.CV_VALUE, BC04_NBI.CV_VALUE)
        ELSE BC04.CV_VALUE
    END AS culvert_cond_rating,

    -- ---------------- grouping [confirmed] ----------------
    DIST.CV_VALUE  AS txdot_district,     -- B.L.04
    NULL           AS climate_zone,       -- [TODO] not an AssetWise field — assign via district/county lookup in Python

    -- ---------------- numeric features ----------------
    STRUCT_LEN.CV_VALUE AS structure_length,      -- [confirmed] B.G.01 NBIS Bridge Length
    MAX_SPAN.CV_VALUE   AS max_span_length,       -- [confirmed] B.G.03
    DECK_WIDTH.CV_VALUE AS deck_width,             -- [confirmed] B.G.05 Bridge Width Out-to-Out
    ROADWAY_W.CV_VALUE  AS roadway_width,          -- [confirmed] B.G.06 Bridge Width Curb-to-Curb
    ADT_CUR.CV_VALUE    AS adt,                    -- [cheatsheet 2026-07-28] B.H.09 = 2306013 (was TODO)
    ADT_YR.CV_VALUE     AS adt_year,               -- [cheatsheet 2026-07-28] B.H.11 = 2306015 (was TODO)
    ADTT.CV_VALUE       AS adt_truck,              -- [cheatsheet 2026-07-28] NEW: B.H.10 = 2306014 truck traffic (Tier 3)
    BW01.CV_VALUE       AS year_built,             -- [confirmed] B.W.01 (was mislabeled "B.W.01" generically in original query)
    NULL AS year_reconstructed,           -- [TODO] B.W work events (2311000) are a repeating group; needs a "latest work year" rollup — confirm structure before wiring
    SKEW.CV_VALUE        AS skew_angle,            -- [confirmed] B.G.11
    NSPANS.CV_VALUE     AS num_spans_main,         -- [cheatsheet 2026-07-28] B.SP.02 instance 1 = 2303003 (main-span count)
    NBEAMS.CV_VALUE     AS num_beam_lines,         -- [cheatsheet 2026-07-28] NEW: B.SP.03 instance 1 = 2303004 (redundancy)
    NULL AS num_spans_approach,           -- [TODO] no single approach-span field in SNBI; would need a rollup across span instances 2..n

    -- ---------------- categorical / structural features ----------------
    OWNER_F.CV_VALUE     AS owner,                 -- [confirmed] B.CL.01
    MAINT_RESP.CV_VALUE  AS maintenance_resp,      -- [confirmed] B.CL.02
    SPAN_MAT.CV_VALUE    AS structure_kind,        -- [cheatsheet 2026-07-28] B.SP.04 Span Material instance 1 = 2303005
    SPAN_TYPE.CV_VALUE   AS structure_type,        -- [cheatsheet 2026-07-28] B.SP.06 Span Type instance 1 = 2303007
    SPAN_CONT.CV_VALUE   AS span_continuity,       -- [cheatsheet 2026-07-28] NEW: B.SP.05 Span Continuity instance 1 = 2303006
    DECK_MAT.CV_VALUE    AS deck_type,             -- [cheatsheet 2026-07-28] B.SP.09 Deck Material and Type instance 1 = 2303010
    WEAR_SURF.CV_VALUE   AS wearing_surface,       -- [cheatsheet 2026-07-28] B.SP.10 Wearing Surface instance 1 = 2303011
    NULL           AS membrane_type,             -- [TODO] NBI 108B Membrane has NO direct SNBI equivalent (abandoned in transition) — leave NULL
    DECK_PROT.CV_VALUE   AS deck_protection,       -- [cheatsheet 2026-07-28] B.SP.11 Deck Protective System instance 1 = 2303012
    DESIGN_LOAD.CV_VALUE AS design_load,           -- [confirmed] B.LR.01
    BG09.CV_VALUE        AS approach_roadway_width, -- [confirmed] B.G.09, reused from original query
    FUNC_CLASS.CV_VALUE  AS functional_class,      -- [cheatsheet 2026-07-28] B.H.01 = 2306005 (was TODO)
    FACIL.CV_VALUE AS facility_carried,          -- [confirmed] NBI FE_ID — reused from original query
    FEAT.CV_VALUE  AS features_intersected,      -- [confirmed] NBI FE_ID — reused from original query

    -- ---------------- NEW condition / exposure features [cheatsheet 2026-07-28] ----------------
    CHAN_COND.CV_VALUE   AS channel_cond_rating,   -- B.C.09 = 2300709 (5th condition rating; scour/substructure signal)
    SCOUR.CV_VALUE       AS scour_vulnerability,   -- B.AP.03 = 2300803 (substructure driver)
    LOAD_POST.CV_VALUE   AS load_posting_status,   -- B.PS.01 = 2310003 (capacity/restriction)
    COUNTY_F.CV_VALUE    AS county_code,           -- B.L.02 = 2300102 (for climate-zone lookup)

    -- ---------------- NEW inspection metadata [cheatsheet 2026-07-28] ----------------
    -- NOTE: TBLCURRENTVALUES holds only the LATEST inspection. The 1996-2023 panel used for
    -- training comes from a history/archive source, not this table. These two fields let you
    -- confirm the real per-inspection date exists; the panel build must read inspection history.
    INSP_DATE.CV_VALUE   AS inspection_date,       -- B.IE.03 Routine = 2302015
    INSP_INT.CV_VALUE    AS inspection_interval    -- B.IE.05 Routine = 2302017

FROM TBLASSETS AST
JOIN tblAssetStatuses STA
    ON AST.ASSET_STATUS_ID = STA.ASSET_STATUS_ID

-- ---------------- confirmed joins (carried over) ----------------
LEFT JOIN TBLCURRENTVALUES BC01     ON AST.AS_ID = BC01.AS_ID     AND BC01.FE_ID     = 2300701
LEFT JOIN TBLCURRENTVALUES BC02     ON AST.AS_ID = BC02.AS_ID     AND BC02.FE_ID     = 2300702
LEFT JOIN TBLCURRENTVALUES BC03     ON AST.AS_ID = BC03.AS_ID     AND BC03.FE_ID     = 2300703
LEFT JOIN TBLCURRENTVALUES BC04     ON AST.AS_ID = BC04.AS_ID     AND BC04.FE_ID     = 2300704
LEFT JOIN TBLCURRENTVALUES BC01_NBI ON AST.AS_ID = BC01_NBI.AS_ID AND BC01_NBI.FE_ID = 2005800
LEFT JOIN TBLCURRENTVALUES BC02_NBI ON AST.AS_ID = BC02_NBI.AS_ID AND BC02_NBI.FE_ID = 2005900
LEFT JOIN TBLCURRENTVALUES BC03_NBI ON AST.AS_ID = BC03_NBI.AS_ID AND BC03_NBI.FE_ID = 2006000
LEFT JOIN TBLCURRENTVALUES BC04_NBI ON AST.AS_ID = BC04_NBI.AS_ID AND BC04_NBI.FE_ID = 2006200
LEFT JOIN TBLCURRENTVALUES DIST     ON AST.AS_ID = DIST.AS_ID     AND DIST.FE_ID     = 2300104
LEFT JOIN TBLCURRENTVALUES FACIL    ON AST.AS_ID = FACIL.AS_ID    AND FACIL.FE_ID    = 2000700
LEFT JOIN TBLCURRENTVALUES FEAT     ON AST.AS_ID = FEAT.AS_ID     AND FEAT.FE_ID     = 2000610

-- confirmed via sql/fe_id_lookup.sql results (2026-07-10)
LEFT JOIN TBLCURRENTVALUES STRUCT_LEN  ON AST.AS_ID = STRUCT_LEN.AS_ID  AND STRUCT_LEN.FE_ID  = 2300401 -- B.G.01 NBIS Bridge Length
LEFT JOIN TBLCURRENTVALUES MAX_SPAN    ON AST.AS_ID = MAX_SPAN.AS_ID    AND MAX_SPAN.FE_ID    = 2300403 -- B.G.03 Maximum Span Length
LEFT JOIN TBLCURRENTVALUES DECK_WIDTH  ON AST.AS_ID = DECK_WIDTH.AS_ID  AND DECK_WIDTH.FE_ID  = 2300405 -- B.G.05 Bridge Width Out-to-Out
LEFT JOIN TBLCURRENTVALUES ROADWAY_W   ON AST.AS_ID = ROADWAY_W.AS_ID   AND ROADWAY_W.FE_ID   = 2300406 -- B.G.06 Bridge Width Curb-to-Curb
LEFT JOIN TBLCURRENTVALUES BW01        ON AST.AS_ID = BW01.AS_ID        AND BW01.FE_ID        = 2300901 -- B.W.01 Year Built
LEFT JOIN TBLCURRENTVALUES SKEW        ON AST.AS_ID = SKEW.AS_ID        AND SKEW.FE_ID        = 2300411 -- B.G.11 Skew
LEFT JOIN TBLCURRENTVALUES OWNER_F     ON AST.AS_ID = OWNER_F.AS_ID     AND OWNER_F.FE_ID     = 2300201 -- B.CL.01 Owner
LEFT JOIN TBLCURRENTVALUES MAINT_RESP  ON AST.AS_ID = MAINT_RESP.AS_ID  AND MAINT_RESP.FE_ID  = 2300202 -- B.CL.02 Maintenance Responsibility
LEFT JOIN TBLCURRENTVALUES DESIGN_LOAD ON AST.AS_ID = DESIGN_LOAD.AS_ID AND DESIGN_LOAD.FE_ID = 2300501 -- B.LR.01 Design Load
LEFT JOIN TBLCURRENTVALUES BG09        ON AST.AS_ID = BG09.AS_ID        AND BG09.FE_ID        = 2300409 -- B.G.09 Approach Roadway Width

-- ---------------- NEW joins, FE_IDs from SNBI Transition CheatSheet (2026-07-28) ----------------
-- Traffic / load (Tier 3)
LEFT JOIN TBLCURRENTVALUES ADT_CUR     ON AST.AS_ID = ADT_CUR.AS_ID     AND ADT_CUR.FE_ID     = 2306013 -- B.H.09 ADT
LEFT JOIN TBLCURRENTVALUES ADT_YR      ON AST.AS_ID = ADT_YR.AS_ID      AND ADT_YR.FE_ID      = 2306015 -- B.H.11 Year of ADT
LEFT JOIN TBLCURRENTVALUES ADTT        ON AST.AS_ID = ADTT.AS_ID        AND ADTT.FE_ID        = 2306014 -- B.H.10 Truck Traffic (ADTT)
LEFT JOIN TBLCURRENTVALUES LOAD_POST   ON AST.AS_ID = LOAD_POST.AS_ID   AND LOAD_POST.FE_ID   = 2310003 -- B.PS.01 Load Posting Status
LEFT JOIN TBLCURRENTVALUES FUNC_CLASS  ON AST.AS_ID = FUNC_CLASS.AS_ID  AND FUNC_CLASS.FE_ID  = 2306005 -- B.H.01 Functional Classification

-- Structural / span attributes, instance 1 = main span (Tier 1; static per bridge — backfill across panel)
LEFT JOIN TBLCURRENTVALUES SPAN_MAT    ON AST.AS_ID = SPAN_MAT.AS_ID    AND SPAN_MAT.FE_ID    = 2303005 -- B.SP.04 Span Material (1)
LEFT JOIN TBLCURRENTVALUES SPAN_TYPE   ON AST.AS_ID = SPAN_TYPE.AS_ID   AND SPAN_TYPE.FE_ID   = 2303007 -- B.SP.06 Span Type (1)
LEFT JOIN TBLCURRENTVALUES SPAN_CONT   ON AST.AS_ID = SPAN_CONT.AS_ID   AND SPAN_CONT.FE_ID   = 2303006 -- B.SP.05 Span Continuity (1)
LEFT JOIN TBLCURRENTVALUES NSPANS      ON AST.AS_ID = NSPANS.AS_ID      AND NSPANS.FE_ID      = 2303003 -- B.SP.02 Number of Spans (1)
LEFT JOIN TBLCURRENTVALUES NBEAMS      ON AST.AS_ID = NBEAMS.AS_ID      AND NBEAMS.FE_ID      = 2303004 -- B.SP.03 Number of Beam Lines (1)
LEFT JOIN TBLCURRENTVALUES DECK_MAT    ON AST.AS_ID = DECK_MAT.AS_ID    AND DECK_MAT.FE_ID    = 2303010 -- B.SP.09 Deck Material and Type (1)
LEFT JOIN TBLCURRENTVALUES WEAR_SURF   ON AST.AS_ID = WEAR_SURF.AS_ID   AND WEAR_SURF.FE_ID   = 2303011 -- B.SP.10 Wearing Surface (1)
LEFT JOIN TBLCURRENTVALUES DECK_PROT   ON AST.AS_ID = DECK_PROT.AS_ID   AND DECK_PROT.FE_ID   = 2303012 -- B.SP.11 Deck Protective System (1)

-- Condition / exposure (Tier 2)
LEFT JOIN TBLCURRENTVALUES CHAN_COND   ON AST.AS_ID = CHAN_COND.AS_ID   AND CHAN_COND.FE_ID   = 2300709 -- B.C.09 Channel Condition Rating
LEFT JOIN TBLCURRENTVALUES SCOUR       ON AST.AS_ID = SCOUR.AS_ID       AND SCOUR.FE_ID       = 2300803 -- B.AP.03 Scour Vulnerability
LEFT JOIN TBLCURRENTVALUES COUNTY_F    ON AST.AS_ID = COUNTY_F.AS_ID    AND COUNTY_F.FE_ID    = 2300102 -- B.L.02 County Code

-- Inspection metadata (latest inspection only in this table)
LEFT JOIN TBLCURRENTVALUES INSP_DATE   ON AST.AS_ID = INSP_DATE.AS_ID   AND INSP_DATE.FE_ID   = 2302015 -- B.IE.03 Inspection Completion Date (Routine)
LEFT JOIN TBLCURRENTVALUES INSP_INT    ON AST.AS_ID = INSP_INT.AS_ID    AND INSP_INT.FE_ID    = 2302017 -- B.IE.05 Inspection Interval (Routine)

-- ---------------- STILL TODO ----------------
-- year_reconstructed: B.W work events (2311000) are a repeating group (max 1 instance as of 12/2024);
--   needs a "most-recent work-completion year" rollup — confirm the work-event table structure first.
-- num_spans_approach: no single SNBI field; would require summing B.SP.02 across span instances 2..n.
-- membrane_type: NBI 108B has no SNBI equivalent (abandoned) — leave NULL.

WHERE AST.AT_ID = 1
AND LEFT(AST.AS_CODE, 1) IN ('0','1','2')
AND AST.AS_CODE <> '0'
AND AST.AS_ROOT = 0
AND LEN(AST.AS_CODE) IN (14, 15)
AND AST.AS_DELETED = 0
AND AST.AS_CODE NOT LIKE '%test%'
AND AST.AS_CODE NOT LIKE '%##%'
AND AST.ASSET_STATUS_ID IN ('0','1','2')
