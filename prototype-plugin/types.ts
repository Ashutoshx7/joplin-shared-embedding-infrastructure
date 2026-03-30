/**
 * Shared API type definitions for the Joplin Embedding & Retrieval Infrastructure
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │  This file is the PUBLIC CONTRACT for cross-plugin usage.   │
 * │  Other AI plugins import these types for type-safe calls    │
 * │  via joplin.commands.execute('aiSearch.*', ...).             │
 * │                                                             │
 * │  Publish as: @joplin/ai-search-types (npm package)          │
 * └─────────────────────────────────────────────────────────────┘
 */

// ─── Core API ──────────────────────────────────────────────────────────

/**
 * The main interface exposed by the Shared Embedding & Retrieval plugin.
 *
 * Consumer plugins access this via Joplin's command system:
 * ```typescript
 * // Query the embedding index
 * const results = await joplin.commands.execute('aiSearch.query', 'my search', { limit: 10 });
 *
 * // Find similar notes
 * const similar = await joplin.commands.execute('aiSearch.findSimilar', noteId, 5);
 *
 * // Get note-level embedding for categorization/clustering
 * const embedding = await joplin.commands.execute('aiSearch.getNoteEmbedding', noteId);
 *
 * // Get all note embeddings for graph/clustering analysis
 * const allEmbs = await joplin.commands.execute('aiSearch.getAllNoteEmbeddings');
 *
 * // Get index stats
 * const stats = await joplin.commands.execute('aiSearch.stats');
 * ```
 */
export interface JoplinEmbeddingAPI {
	/**
	 * Index or re-index a single note.
	 * @shikuz's core interface: put(note).
	 */
	put(noteId: string): Promise<void>;

	/**
	 * Hybrid search: vector (semantic) + keyword (FTS4), fused via RRF.
	 * Returns ranked results with snippets, heading paths, and scores.
	 * Supports query decomposition and cross-encoder reranking.
	 */
	query(text: string, options?: QueryOptions): Promise<SearchResult[]>;

	/**
	 * Embed arbitrary text using the configured embedding model.
	 * Used by chat/RAG plugins for prompt engineering and context assembly.
	 */
	embed(text: string): Promise<number[]>;

	/**
	 * Find notes semantically similar to a given note.
	 * Uses note-level embeddings (mean of chunk vectors).
	 */
	findSimilarNotes(noteId: string, k?: number): Promise<SearchResult[]>;

	/**
	 * Get the note-level embedding vector for a specific note.
	 * Useful for categorization (k-means clustering), note graphs
	 * (embedding distance analysis), and auto-tagging.
	 * Returns null if note is not indexed.
	 */
	getNoteEmbedding(noteId: string): Promise<number[] | null>;

	/**
	 * Get all note-level embeddings for batch analysis.
	 * Useful for building note graphs and clustering.
	 */
	getAllNoteEmbeddings(): Promise<{ noteId: string; embedding: number[] }[]>;

	/**
	 * Get chunk-level embedding vectors for a specific note.
	 * Useful for fine-grained analysis within a note.
	 */
	getChunkEmbeddings(noteId: string): Promise<number[][]>;

	/**
	 * Get index statistics (note count, chunk count, model, disk size).
	 */
	getStats(): Promise<IndexStats>;

	/**
	 * Rebuild the entire embedding index from scratch.
	 * Warning: This is expensive — ~5 min for 2000 notes.
	 */
	reindexAll(): Promise<void>;

	/**
	 * Check if the index is ready for queries.
	 */
	isReady(): boolean;
}

// ─── Query Options ─────────────────────────────────────────────────────

export interface QueryOptions {
	/** Maximum number of results. Default: 10. */
	limit?: number;

	/** Combine vector + keyword search via RRF. Default: true. */
	hybrid?: boolean;

	/**
	 * Hybrid balance: 0.0 = keyword only, 1.0 = vector only, 0.5 = equal.
	 * Default: 0.5 (reads from user settings if not specified).
	 * @adamoutler @shikuz: "hybrid search balance slider"
	 */
	hybridBalance?: number;

	/** Filter by notebook ID. */
	notebookId?: string;

	/** Enable cross-encoder reranking of top results. */
	rerank?: boolean;

	/** Enable query decomposition for complex queries. */
	decompose?: boolean;
}

// ─── Results ───────────────────────────────────────────────────────────

export interface SearchResult {
	/** Joplin note ID. */
	noteId: string;

	/** Note title. */
	noteTitle: string;

	/** RSE segment — coherent passage from merged adjacent chunks. */
	snippet: string;

	/** Heading breadcrumb, e.g. "Project > Backend > Database". */
	headingPath: string;

	/** RRF fusion score (rank-based, not a probability). */
	score: number;

	/** Where this result came from: vector-only, keyword-only, or both. */
	source: 'vector' | 'keyword' | 'hybrid';
}

// ─── Index Stats ───────────────────────────────────────────────────────

export interface IndexStats {
	/** Number of distinct notes in the index. */
	totalNotes: number;

	/** Total number of chunks across all notes. */
	totalChunks: number;

	/** Name of the embedding model used. */
	modelName: string;

	/** Embedding vector dimensionality. */
	dimensions: number;

	/** Timestamp of the most recent indexing operation. */
	lastUpdated: number;

	/** Size of the sql.js database on disk in bytes. */
	dbSizeBytes: number;
}

// ─── Consumer Usage Examples ───────────────────────────────────────────

/**
 * Example: AI Chat plugin using the shared infrastructure for RAG
 *
 * ```typescript
 * // In your chat plugin's onStart():
 * async function getChatContext(userMessage: string): Promise<string> {
 *   const results = await joplin.commands.execute('aiSearch.query', userMessage, {
 *     limit: 5,
 *     hybrid: true,
 *     rerank: true,     // Enable precision reranking for chat context
 *   });
 *
 *   if (!results || results.length === 0) return '';
 *
 *   return results.map(r =>
 *     `--- Note: ${r.noteTitle} (${r.headingPath}) ---\n${r.snippet}`
 *   ).join('\n\n');
 * }
 * ```
 *
 * Example: Categorization plugin using note embeddings for clustering
 *
 * ```typescript
 * // Find clusters of related notes using k-means
 * async function clusterNotes(): Promise<Map<number, string[]>> {
 *   const allEmbs = await joplin.commands.execute('aiSearch.getAllNoteEmbeddings');
 *   // Run k-means on the embedding vectors
 *   return kMeans(allEmbs, k=10);
 * }
 * ```
 *
 * Example: Note graph plugin using similarity distances
 *
 * ```typescript
 * async function buildSimilarityGraph(): Promise<Edge[]> {
 *   const allEmbs = await joplin.commands.execute('aiSearch.getAllNoteEmbeddings');
 *   const edges = [];
 *   for (let i = 0; i < allEmbs.length; i++) {
 *     for (let j = i + 1; j < allEmbs.length; j++) {
 *       const sim = cosineSimilarity(allEmbs[i].embedding, allEmbs[j].embedding);
 *       if (sim > 0.7) edges.push({ from: allEmbs[i].noteId, to: allEmbs[j].noteId, weight: sim });
 *     }
 *   }
 *   return edges;
 * }
 * ```
 */
export type _ConsumerExamples = never; // Type-only, not used at runtime
