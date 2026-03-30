import Logger from '@joplin/utils/Logger';
import Note from '../../models/Note';
import { NoteEntity } from '../database/types';
import { ChunkingEngine } from './ChunkingEngine';
import { EmbeddingProvider, createEmbeddingProvider, ProviderType } from './EmbeddingProvider';
import { VectorStore, ScoredChunk, IndexStats } from './VectorStore';
import { RetrievalEngine, SearchResult, SearchOptions as RetrievalSearchOptions } from './RetrievalEngine';

const logger = Logger.create('EmbeddingService');

export interface EmbeddingServiceOptions {
	profileDir: string;
	enabled?: boolean;
	provider?: ProviderType;
	ollamaEndpoint?: string;
	ollamaModel?: string;
	apiKey?: string;
}

/**
 * Core Embedding Service for Joplin
 *
 * Singleton service living in packages/lib/services/embedding/
 * Provides vector indexing and hybrid retrieval for all AI features.
 *
 * Consumers: AI Search, Chat, Categorisation, Note Graphs
 */
export default class EmbeddingService {

	public static instance_: EmbeddingService = null;

	private db_: any = null;
	private provider_: EmbeddingProvider = null;
	private vectorStore_: VectorStore = null;
	private chunker_: ChunkingEngine = null;
	private retrieval_: RetrievalEngine = null;
	private profileDir_: string = '';
	private enabled_: boolean = false;
	private indexing_: boolean = false;
	private disposed_: boolean = false;
	private syncTimer_: any = null;
	private logger_: any = logger;

	public static instance(): EmbeddingService {
		if (!this.instance_) {
			this.instance_ = new EmbeddingService();
		}
		return this.instance_;
	}

	public setDb(db: any) {
		this.db_ = db;
	}

	public setLogger(l: any) {
		this.logger_ = l;
	}

	/**
	 * Initialize the embedding service.
	 * Called from BaseApplication after database is ready.
	 */
	public async initialize(options: EmbeddingServiceOptions): Promise<void> {
		this.profileDir_ = options.profileDir;
		this.enabled_ = options.enabled !== false;

		if (!this.enabled_) {
			this.logger_.info('EmbeddingService: disabled in settings');
			return;
		}

		this.logger_.info('EmbeddingService: initializing...');

		try {
			// 1. Create chunker
			this.chunker_ = new ChunkingEngine();

			// 2. Create embedding provider
			const providerType = options.provider || 'ollama';
			this.provider_ = createEmbeddingProvider(providerType, {
				endpoint: options.ollamaEndpoint || 'http://localhost:11434',
				modelId: options.ollamaModel || 'nomic-embed-text',
				apiKey: options.apiKey,
			});

			// 3. Create vector store
			const dbPath = `${this.profileDir_}/embedding-index.sqlite`;
			this.vectorStore_ = new VectorStore(dbPath);
			await this.vectorStore_.init();
			this.vectorStore_.setMeta('model_name', this.provider_.name);

			// 4. Create retrieval engine
			this.retrieval_ = new RetrievalEngine(this.vectorStore_, this.provider_);

			this.logger_.info(`EmbeddingService: ready (provider=${providerType}, dims=${this.provider_.dimensions})`);
		} catch (e) {
			this.logger_.error('EmbeddingService: initialization failed:', e);
			this.enabled_ = false;
		}
	}

	public isReady(): boolean {
		return this.enabled_ && this.vectorStore_ !== null && this.provider_ !== null;
	}

	// ─── Public API (consumed by plugins via EmbeddingService.instance()) ────

	/**
	 * put(noteId) — index/re-index a single note.
	 * @shikuz's core interface.
	 */
	public async put(noteId: string): Promise<void> {
		if (!this.isReady()) return;
		const note = await Note.load(noteId, { fields: ['id', 'title', 'body', 'parent_id'] });
		if (note) await this.indexNote(note);
	}

	/**
	 * Semantic + hybrid search over all indexed notes.
	 */
	public async search(query: string, options: RetrievalSearchOptions = {}): Promise<SearchResult[]> {
		if (!this.isReady()) return [];
		return this.retrieval_.search(query, options, this.db_);
	}

	/**
	 * Embed arbitrary text — for RAG/chat context assembly.
	 */
	public async embed(text: string): Promise<number[]> {
		if (!this.isReady()) return [];
		const embedding = await this.provider_.embed(text);
		return Array.from(embedding);
	}

	/**
	 * Find notes semantically similar to a given note.
	 */
	public findSimilarNotes(noteId: string, limit: number = 5): ScoredChunk[] {
		if (!this.isReady()) return [];
		return this.vectorStore_.findSimilarNotes(noteId, limit);
	}

	/**
	 * Get note-level embedding vector.
	 * Used by categorization (k-means), note graphs (distance analysis).
	 */
	public getNoteEmbedding(noteId: string): number[] | null {
		if (!this.isReady()) return null;
		return this.vectorStore_.getNoteEmbedding(noteId);
	}

	/**
	 * Get all note-level embeddings for batch analysis.
	 */
	public getAllNoteEmbeddings(): { noteId: string; embedding: number[] }[] {
		if (!this.isReady()) return [];
		return this.vectorStore_.getAllNoteEmbeddings();
	}

	/**
	 * Get chunk-level embeddings for a specific note.
	 */
	public getChunkEmbeddings(noteId: string): number[][] {
		if (!this.isReady()) return [];
		return this.vectorStore_.getChunkEmbeddings(noteId);
	}

	/**
	 * Get index statistics.
	 */
	public getStats(): IndexStats {
		if (!this.isReady()) {
			return { totalNotes: 0, totalChunks: 0, modelName: '', lastUpdated: 0, dbSizeBytes: 0 };
		}
		return this.vectorStore_.getStats();
	}

	// ─── Indexing ─────────────────────────────────────────────────────────

	/**
	 * Index a single note (chunk + embed + store).
	 */
	public async indexNote(note: NoteEntity): Promise<void> {
		if (!this.isReady() || !note.body) return;

		const contentHash = this.chunker_.computeHash(note.body);

		// Skip if unchanged
		if (this.vectorStore_.isNoteIndexed(note.id, contentHash)) {
			return;
		}

		// Chunk
		const chunks = this.chunker_.chunkNote(note.id, note.title || '', note.body);
		if (chunks.length === 0) return;

		// Embed
		const texts = chunks.map((c: { text: string }) => c.text);
		const embeddings = await this.provider_.embedBatch(texts);

		// Store
		this.vectorStore_.storeNoteChunks(chunks, embeddings);
	}

	/**
	 * Build index for all notes. Reports progress via callback.
	 */
	public async buildIndex(onProgress?: (current: number, total: number) => void): Promise<void> {
		if (!this.isReady()) throw new Error('EmbeddingService not initialized');
		if (this.indexing_) throw new Error('Indexing already in progress');

		this.indexing_ = true;
		try {
			// Get all non-deleted, non-conflict notes
			const notes: NoteEntity[] = await Note.all({
				fields: ['id', 'title', 'body', 'updated_time', 'is_conflict'],
				order: [{ by: 'updated_time', dir: 'DESC' }],
			});

			const validNotes = notes.filter(n => !n.is_conflict && n.body);
			const total = validNotes.length;

			this.logger_.info(`EmbeddingService: indexing ${total} notes...`);

			for (let i = 0; i < validNotes.length; i++) {
				if (this.disposed_) break;

				const note = validNotes[i];
				try {
					await this.indexNote(note);
				} catch (e) {
					this.logger_.warn(`EmbeddingService: failed to index note ${note.id}:`, e);
				}

				if (onProgress) onProgress(i + 1, total);

				// Yield to event loop every 5 notes
				if (i % 5 === 0) await new Promise(r => setTimeout(r, 0));
			}

			await this.vectorStore_.flushToDisk();
			this.logger_.info(`EmbeddingService: indexing complete (${total} notes)`);
		} finally {
			this.indexing_ = false;
		}
	}

	/**
	 * Clear the entire index.
	 */
	public async clearIndex(): Promise<void> {
		if (this.vectorStore_) {
			this.vectorStore_.clear();
			await this.vectorStore_.flushToDisk();
		}
	}

	// ─── Lifecycle ────────────────────────────────────────────────────────

	public async destroy(): Promise<void> {
		this.disposed_ = true;
		if (this.syncTimer_) {
			clearInterval(this.syncTimer_);
			this.syncTimer_ = null;
		}
		if (this.provider_) {
			this.provider_.dispose();
			this.provider_ = null;
		}
		if (this.vectorStore_) {
			await this.vectorStore_.close();
			this.vectorStore_ = null;
		}
	}
}
