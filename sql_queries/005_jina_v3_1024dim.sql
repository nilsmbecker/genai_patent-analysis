-- ============================================================
-- 005_jina_v3_1024dim.sql
-- Migrates embedding dimension from 384 (BAAI/bge-small-en-v1.5)
-- to 1024 (jinaai/jina-embeddings-v3).
--
-- IMPORTANT: Run this after merging the Jina v3 branch.
-- All existing patent data is deleted because the stored 384-dim
-- embeddings are incompatible with the new model. Re-ingest all
-- patents after running this migration.
-- ============================================================

-- Step 1: Delete all ingested data.
-- Deleting patent_documents cascades to patent_chunks and patent_images.
DELETE FROM patent_documents;

-- Step 2: Drop the HNSW index (bound to VECTOR(384)).
DROP INDEX IF EXISTS idx_patent_chunks_embedding_hnsw;

-- Step 3: Change the embedding column to 1024 dimensions.
-- All rows were deleted above so no data conversion is needed.
ALTER TABLE patent_chunks
    ALTER COLUMN embedding TYPE VECTOR(1024);

-- Step 4: Recreate the HNSW index for 1024-dim vectors.
CREATE INDEX idx_patent_chunks_embedding_hnsw
    ON patent_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Step 5: Replace the hybrid search function with a 1024-dim signature.
CREATE OR REPLACE FUNCTION match_patent_hybrid(
    query_embedding     VECTOR(1024),
    query_text          TEXT,
    filter_jurisdiction TEXT    DEFAULT NULL,
    match_count         INT     DEFAULT 3,
    rrf_k               INT     DEFAULT 60
)
RETURNS TABLE (
    chunk_id      UUID,
    patent_id     UUID,
    patent_number TEXT,
    title         TEXT,
    jurisdiction  TEXT,
    section_type  TEXT,
    content       TEXT,
    fts_rank      FLOAT8,
    vector_rank   FLOAT8,
    rrf_score     FLOAT8
)
LANGUAGE plpgsql STABLE AS $$
BEGIN
    RETURN QUERY
    WITH
    fts_ranked AS (
        SELECT
            pc.id                                                          AS chunk_id,
            pc.patent_id,
            pd.patent_number,
            pd.title,
            pd.jurisdiction,
            pc.section_type,
            pc.content,
            ts_rank_cd(pc.fts_tokens, plainto_tsquery('english', query_text))::FLOAT8 AS rank_score,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(pc.fts_tokens, plainto_tsquery('english', query_text)) DESC
            ) AS row_num
        FROM patent_chunks pc
        JOIN patent_documents pd ON pd.id = pc.patent_id
        WHERE (filter_jurisdiction IS NULL OR pd.jurisdiction = filter_jurisdiction)
          AND pc.fts_tokens IS NOT NULL
          AND pc.fts_tokens @@ plainto_tsquery('english', query_text)
        ORDER BY rank_score DESC
        LIMIT match_count * 5
    ),
    vec_ranked AS (
        SELECT
            pc.id                                    AS chunk_id,
            pc.patent_id,
            pd.patent_number,
            pd.title,
            pd.jurisdiction,
            pc.section_type,
            pc.content,
            (1 - (pc.embedding <=> query_embedding))::FLOAT8 AS rank_score,
            ROW_NUMBER() OVER (
                ORDER BY pc.embedding <=> query_embedding ASC
            ) AS row_num
        FROM patent_chunks pc
        JOIN patent_documents pd ON pd.id = pc.patent_id
        WHERE (filter_jurisdiction IS NULL OR pd.jurisdiction = filter_jurisdiction)
        ORDER BY pc.embedding <=> query_embedding ASC
        LIMIT match_count * 5
    ),
    fused AS (
        SELECT
            COALESCE(f.chunk_id,      v.chunk_id)      AS chunk_id,
            COALESCE(f.patent_id,     v.patent_id)     AS patent_id,
            COALESCE(f.patent_number, v.patent_number) AS patent_number,
            COALESCE(f.title,         v.title)         AS title,
            COALESCE(f.jurisdiction,  v.jurisdiction)  AS jurisdiction,
            COALESCE(f.section_type,  v.section_type)  AS section_type,
            COALESCE(f.content,       v.content)       AS content,
            COALESCE(f.rank_score, 0.0)                AS fts_rank,
            COALESCE(v.rank_score, 0.0)                AS vector_rank,
            (
                COALESCE(1.0 / (rrf_k + f.row_num), 0.0) +
                COALESCE(1.0 / (rrf_k + v.row_num), 0.0)
            )::FLOAT8                                  AS rrf_score
        FROM fts_ranked f
        FULL OUTER JOIN vec_ranked v ON f.chunk_id = v.chunk_id
    )
    SELECT
        fused.chunk_id, fused.patent_id, fused.patent_number, fused.title,
        fused.jurisdiction, fused.section_type, fused.content,
        fused.fts_rank, fused.vector_rank, fused.rrf_score
    FROM fused
    ORDER BY fused.rrf_score DESC
    LIMIT match_count;
END;
$$;
