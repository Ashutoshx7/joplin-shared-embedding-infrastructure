/**
 * Hybrid retrieval engine
 *
 * Combines vector search (semantic) with Joplin's FTS4 (lexical) using
 * Reciprocal Rank Fusion (RRF). Includes:
 *  - Relevant Segment Extraction (RSE) for coherent passages
 *  - Cross-encoder reranking for precision
 *  - Query decomposition for complex queries
 *  - Hybrid balance slider (0=keyword, 1=vector)
 *  - Notebook-scoped filtering
 */

import { ScoredChunk, VectorStore } from './vectorStore';
import { EmbeddingProvider } from './embeddings';
import { Reranker, RerankerConfig } from './reranker';
import { QueryDecomposer, QueryDecomposerConfig } from './queryDecomposer';

export interface SearchResult {
	noteId: string;
	noteTitle: string;
	snippet: string;       // RSE segment or best chunk text
	headingPath: string;   // "Project > Backend > Database"
	score: number;
	source: 'vector' | 'keyword' | 'hybrid';
}

export interface SearchOptions {
	/** Maximum number of results. Default: 10. */
	limit?: number;
	/** Combine vector + keyword search via RRF. Default: true. */
	hybrid?: boolean;
	/**
	 * Hybrid balance: 0.0 = keyword only, 1.0 = vector only, 0.5 = equal.
	 * Default: 0.5.
	 * @adamoutler: "hybrid search balance slider"
	 */
	hybridBalance?: number;
	/** Filter by notebook ID. */
	notebookId?: string;
	/** Enable reranking of top results. Default: use reranker.enabled. */
	rerank?: boolean;
	/** Enable query decomposition. Default: use decomposer.enabled. */
	decompose?: boolean;
}

export interface KeywordResult {
	id: string;
	title: string;
	body?: string;
}

type JoplinDataGet = (path: string[], query?: any) => Promise<any>;

const RRF_K = 60; // Standard RRF constant from Cormack et al., SIGIR 2009

/**
 * Reciprocal Rank Fusion — merges two ranked lists by rank position.
 * Score = Σ 1/(k + rank_i), where k=60 is universally good.
 *
 * @param vectorWeight - Weight for vector results (0-1)
 * @param keywordWeight - Weight for keyword results (0-1)
 */
function reciprocalRankFusion(
	vectorChunks: ScoredChunk[],
	keywordNoteIds: string[],
	k: number = RRF_K,
	vectorWeight: number = 1.0,
	keywordWeight: number = 1.0,
): Map<string, { score: number; chunks: ScoredChunk[]; source: 'vector' | 'keyword' | 'hybrid' }> {

	const noteScores = new Map<string, {
		score: number;
		chunks: ScoredChunk[];
		source: 'vector' | 'keyword' | 'hybrid';
	}>();

	// Score from vector results (chunk-level → aggregate to note)
	const noteFromVector = new Set<string>();
	vectorChunks.forEach((chunk, rank) => {
		const existing = noteScores.get(chunk.noteId);
		const rrfScore = (1 / (k + rank + 1)) * vectorWeight;

		if (existing) {
			existing.score += rrfScore;
			existing.chunks.push(chunk);
		} else {
			noteScores.set(chunk.noteId, {
				score: rrfScore,
				chunks: [chunk],
				source: 'vector',
			});
		}
		noteFromVector.add(chunk.noteId);
	});

	// Score from keyword results (note-level)
	keywordNoteIds.forEach((noteId, rank) => {
		const rrfScore = (1 / (k + rank + 1)) * keywordWeight;
		const existing = noteScores.get(noteId);

		if (existing) {
			existing.score += rrfScore;
			existing.source = 'hybrid'; // Appears in both lists
		} else {
			noteScores.set(noteId, {
				score: rrfScore,
				chunks: [],
				source: 'keyword',
			});
		}
	});

	return noteScores;
}

/**
 * Relevant Segment Extraction (RSE) — merge adjacent high-scoring chunks
 * from the same note into coherent passages. Also fills in "sandwiched"
 * low-scoring chunks between two relevant ones (gap ≤ 2).
 */
function extractRelevantSegments(chunks: ScoredChunk[]): string {
	if (chunks.length === 0) return '';
	if (chunks.length === 1) return truncateSnippet(chunks[0].text);

	// Sort by chunk index for sequential merging
	const sorted = [...chunks].sort((a, b) => a.chunkIndex - b.chunkIndex);

	const segments: string[] = [];
	let current = [sorted[0]];

	for (let i = 1; i < sorted.length; i++) {
		const gap = sorted[i].chunkIndex - current[current.length - 1].chunkIndex;

		if (gap <= 2) {
			// Adjacent or sandwiched — merge
			current.push(sorted[i]);
		} else {
			// Non-adjacent — flush segment and start new one
			segments.push(mergeChunkTexts(current));
			current = [sorted[i]];
		}
	}
	segments.push(mergeChunkTexts(current));

	return truncateSnippet(segments.join('\n\n---\n\n'));
}

function mergeChunkTexts(chunks: ScoredChunk[]): string {
	return chunks.map(c => c.text).join('\n\n');
}

function truncateSnippet(text: string, maxChars: number = 500): string {
	if (text.length <= maxChars) return text;
	// Truncate at word boundary
	const truncated = text.substring(0, maxChars);
	const lastSpace = truncated.lastIndexOf(' ');
	return (lastSpace > maxChars * 0.8 ? truncated.substring(0, lastSpace) : truncated) + '…';
}

/**
 * Main retrieval engine — orchestrates hybrid search with reranking
 * and query decomposition.
 */
export class RetrievalEngine {
	private vectorStore: VectorStore;
	private embeddingProvider: EmbeddingProvider;
	private joplinDataGet: JoplinDataGet;
	private reranker: Reranker | null;
	private decomposer: QueryDecomposer | null;

	constructor(
		vectorStore: VectorStore,
		embeddingProvider: EmbeddingProvider,
		joplinDataGet: JoplinDataGet,
		reranker?: Reranker,
		decomposer?: QueryDecomposer,
	) {
		this.vectorStore = vectorStore;
		this.embeddingProvider = embeddingProvider;
		this.joplinDataGet = joplinDataGet;
		this.reranker = reranker || null;
		this.decomposer = decomposer || null;
	}

	/**
	 * Hybrid search with full pipeline:
	 *  1. Query decomposition (if enabled)
	 *  2. Vector + keyword search per sub-query
	 *  3. RRF fusion with hybrid balance
	 *  4. Cross-encoder reranking (if enabled)
	 *  5. RSE segment extraction
	 */
	async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
		const limit = options.limit ?? 10;
		const hybrid = options.hybrid ?? true;
		const hybridBalance = options.hybridBalance ?? 0.5;
		const shouldRerank = options.rerank ?? (this.reranker?.enabled ?? false);
		const shouldDecompose = options.decompose ?? (this.decomposer?.enabled ?? false);

		// Weights from hybrid balance: 0=keyword only, 1=vector only
		const vectorWeight = hybridBalance;
		const keywordWeight = 1 - hybridBalance;

		// Step 1: Query decomposition
		let subQueries = [query];
		if (shouldDecompose && this.decomposer) {
			const decomposed = await this.decomposer.decompose(query);
			subQueries = decomposed.subQueries;
			console.info(`AI Search: decomposed into ${subQueries.length} sub-queries (${decomposed.strategy})`);
		}

		// Step 2: Retrieve per sub-query and merge
		const allVectorChunks: ScoredChunk[] = [];
		const allKeywordNoteIds: string[] = [];
		const seenKeyword = new Set<string>();

		for (const subQuery of subQueries) {
			// Vector search
			const queryEmb = await this.embeddingProvider.embedForQuery(subQuery);
			const vectorResults = this.vectorStore.searchByVector(queryEmb, 50, options.notebookId);
			allVectorChunks.push(...vectorResults);

			// Keyword search
			if (hybrid && keywordWeight > 0) {
				try {
					const keywordResults = await this.joplinDataGet(
						['search'],
						{
							query: subQuery,
							fields: 'id,title',
							limit: 50,
						},
					);

					if (keywordResults && keywordResults.items) {
						for (const r of keywordResults.items) {
							if (!seenKeyword.has(r.id)) {
								seenKeyword.add(r.id);
								allKeywordNoteIds.push(r.id);
							}
						}
					}
				} catch (e) {
					console.warn('Keyword search failed, using vector-only:', e);
				}
			}
		}

		// Step 3: RRF fusion with weighted balance
		const fusedScores = reciprocalRankFusion(
			allVectorChunks, allKeywordNoteIds, RRF_K,
			vectorWeight, keywordWeight,
		);

		// Step 4: Build initial sorted list
		let sortedEntries = [...fusedScores.entries()]
			.sort((a, b) => b[1].score - a[1].score)
			.filter(([_, data]) => data.score > 0.001);

		// Step 5: Reranking (on top candidates)
		if (shouldRerank && this.reranker && this.reranker.enabled) {
			// Flatten top entries into chunks for reranking
			const topChunks: ScoredChunk[] = [];
			for (const [_, data] of sortedEntries.slice(0, 20)) {
				if (data.chunks.length > 0) {
					// Take best chunk per note for reranking
					const bestChunk = data.chunks.reduce(
						(best, c) => c.score > best.score ? c : best,
						data.chunks[0],
					);
					topChunks.push(bestChunk);
				}
			}

			if (topChunks.length > 0) {
				const reranked = await this.reranker.rerank(query, topChunks);
				// Re-order sortedEntries based on reranked order
				const rerankedNoteOrder = new Map<string, number>();
				reranked.forEach((chunk, idx) => {
					if (!rerankedNoteOrder.has(chunk.noteId)) {
						rerankedNoteOrder.set(chunk.noteId, chunk.score);
					}
				});

				sortedEntries.sort((a, b) => {
					const aScore = rerankedNoteOrder.get(a[0]) ?? a[1].score;
					const bScore = rerankedNoteOrder.get(b[0]) ?? b[1].score;
					return bScore - aScore;
				});
			}
		}

		// Step 6: Build results with RSE segments
		const results: SearchResult[] = [];

		for (const [noteId, data] of sortedEntries.slice(0, limit)) {
			const snippet = data.chunks.length > 0
				? extractRelevantSegments(data.chunks)
				: '';

			const headingPath = data.chunks.length > 0
				? data.chunks[0].headingPath
				: '';

			const noteTitle = data.chunks.length > 0
				? data.chunks[0].noteTitle
				: noteId; // Fallback

			results.push({
				noteId,
				noteTitle,
				snippet,
				headingPath,
				score: data.score,
				source: data.source,
			});
		}

		return results;
	}

	/**
	 * Find notes similar to a given note.
	 */
	async findSimilarNotes(noteId: string, limit: number = 5): Promise<SearchResult[]> {
		const similar = this.vectorStore.findSimilarNotes(noteId, limit);

		// Enrich with note titles
		const results: SearchResult[] = [];
		for (const chunk of similar) {
			let title = chunk.noteTitle;
			if (!title) {
				try {
					const note = await this.joplinDataGet(
						['notes', chunk.noteId],
						{ fields: 'title' },
					);
					title = note?.title || chunk.noteId;
				} catch {
					title = chunk.noteId;
				}
			}

			results.push({
				noteId: chunk.noteId,
				noteTitle: title,
				snippet: '',
				headingPath: '',
				score: chunk.score,
				source: 'vector',
			});
		}

		return results;
	}
}
