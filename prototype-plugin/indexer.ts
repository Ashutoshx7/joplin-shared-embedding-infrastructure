/**
 * Incremental indexer
 *
 * 3-tier change detection:
 *  1. onNoteChange() — immediate re-index of active note (debounced)
 *  2. onSyncComplete() — diff all notes by updated_time
 *  3. Periodic /events API polling — cursor-based catchall
 *
 * Cold start indexes recent notes first for fast time-to-first-search.
 */

import { chunkNote, NoteChunk } from './chunker';
import { EmbeddingProvider } from './embeddings';
import { VectorStore } from './vectorStore';

type JoplinDataGet = (path: string[], query?: any) => Promise<any>;

export interface IndexProgress {
	phase: 'idle' | 'indexing' | 'ready';
	current: number;
	total: number;
	message: string;
}

type ProgressCallback = (progress: IndexProgress) => void;

/**
 * Simple djb2 hash for comparing note body content.
 */
function contentHash(text: string): string {
	let hash = 5381;
	for (let i = 0; i < text.length; i++) {
		hash = ((hash << 5) + hash + text.charCodeAt(i)) & 0xffffffff;
	}
	return hash.toString(16).padStart(8, '0');
}

export class Indexer {
	private vectorStore: VectorStore;
	private embeddingProvider: EmbeddingProvider;
	private joplinDataGet: JoplinDataGet;
	private onProgress: ProgressCallback;

	private isRunning = false;
	private cancelled = false;
	private eventsCursor: string = '0';

	constructor(
		vectorStore: VectorStore,
		embeddingProvider: EmbeddingProvider,
		joplinDataGet: JoplinDataGet,
		onProgress: ProgressCallback = () => {},
	) {
		this.vectorStore = vectorStore;
		this.embeddingProvider = embeddingProvider;
		this.joplinDataGet = joplinDataGet;
		this.onProgress = onProgress;
	}

	/**
	 * Full index of all notes. Indexes recent notes first for fast
	 * time-to-first-search. Yields to event loop between batches.
	 */
	async indexAll(): Promise<void> {
		if (this.isRunning) return;
		this.isRunning = true;
		this.cancelled = false;

		try {
			// Fetch all note IDs, ordered by updated_time DESC (recent first)
			const allNotes = await this.fetchAllNoteIds();
			const total = allNotes.length;

			this.onProgress({
				phase: 'indexing',
				current: 0,
				total,
				message: `Indexing ${total} notes…`,
			});

			const indexedIds = this.vectorStore.getIndexedNoteIds();

			for (let i = 0; i < allNotes.length; i++) {
				if (this.cancelled) break;

				const note = allNotes[i];

				// Skip if already indexed with same content
				const hash = contentHash(note.body || '');
				if (indexedIds.has(note.id) && this.vectorStore.isNoteIndexed(note.id, hash)) {
					continue;
				}

				await this.indexSingleNote(note);

				// Update progress
				if ((i + 1) % 5 === 0 || i === allNotes.length - 1) {
					this.onProgress({
						phase: 'indexing',
						current: i + 1,
						total,
						message: `Indexed ${i + 1} of ${total} notes`,
					});
				}

				// Yield to event loop every 10 notes to keep Joplin responsive
				if ((i + 1) % 10 === 0) {
					await new Promise(resolve => setTimeout(resolve, 0));
				}
			}

			// Initialize events cursor for incremental updates
			await this.initEventsCursor();

			this.onProgress({
				phase: 'ready',
				current: total,
				total,
				message: `Index ready — ${total} notes indexed`,
			});
		} finally {
			this.isRunning = false;
		}
	}

	/**
	 * Re-index a single note by ID (used by onNoteChange).
	 */
	async reindexNote(noteId: string): Promise<void> {
		try {
			const note = await this.joplinDataGet(
				['notes', noteId],
				{ fields: 'id,title,body,parent_id,is_conflict,encryption_applied,deleted_time' },
			);

			if (!note || note.is_conflict || note.encryption_applied || note.deleted_time) {
				this.vectorStore.deleteNote(noteId);
				return;
			}

			await this.indexSingleNote(note);
		} catch (e) {
			console.warn(`Indexer: Failed to re-index note ${noteId}:`, e);
		}
	}

	/**
	 * Delete a note from the index.
	 */
	deleteFromIndex(noteId: string): void {
		this.vectorStore.deleteNote(noteId);
	}

	/**
	 * Process changes from the Events API (cursor-based polling).
	 * Returns the number of changes processed.
	 */
	async processEvents(): Promise<number> {
		let processed = 0;

		try {
			let hasMore = true;

			while (hasMore) {
				const response = await this.joplinDataGet(
					['events'],
					{ cursor: this.eventsCursor },
				);

				if (!response || !response.items) break;

				for (const event of response.items) {
					// Only process note changes (item_type 1 = Note)
					if (event.item_type !== 1) continue;

					if (event.type === 1 || event.type === 2) {
						// Created or Updated
						await this.reindexNote(event.item_id);
						processed++;
					} else if (event.type === 3) {
						// Deleted
						this.deleteFromIndex(event.item_id);
						processed++;
					}
				}

				this.eventsCursor = response.cursor;
				hasMore = response.has_more;
			}
		} catch (e) {
			console.warn('Indexer: Events polling failed:', e);
		}

		return processed;
	}

	/**
	 * Cancel a running indexing operation.
	 */
	cancel(): void {
		this.cancelled = true;
	}

	get running(): boolean {
		return this.isRunning;
	}

	// ─── Private ────────────────────────────────────────────────────────

	private async indexSingleNote(note: any): Promise<void> {
		try {
			const chunks = chunkNote(
				note.id,
				note.title || '',
				note.body || '',
				note.parent_id || '',
			);

			if (chunks.length === 0) return;

			// Embed all chunks
			const embeddings = await this.embeddingProvider.embedBatch(
				chunks.map(c => c.text),
			);

			// Store in vector DB
			this.vectorStore.storeNoteChunks(chunks, embeddings);
		} catch (e) {
			console.warn(`Indexer: Failed to index note ${note.id}:`, e);
		}
	}

	private async fetchAllNoteIds(): Promise<any[]> {
		const notes: any[] = [];
		let page = 1;
		let hasMore = true;

		while (hasMore) {
			const response = await this.joplinDataGet(
				['notes'],
				{
					fields: 'id,title,body,parent_id,is_conflict,encryption_applied,deleted_time',
					order_by: 'user_updated_time',
					order_dir: 'DESC',
					limit: 100,
					page: page,
				},
			);

			if (!response || !response.items || response.items.length === 0) break;

			// Filter out conflicts, encrypted, and deleted notes
			const valid = response.items.filter((n: any) =>
				!n.is_conflict && !n.encryption_applied && !n.deleted_time,
			);
			notes.push(...valid);

			hasMore = response.has_more;
			page++;
		}

		return notes;
	}

	private async initEventsCursor(): Promise<void> {
		try {
			// Get current cursor position (latest change ID)
			const response = await this.joplinDataGet(['events'], {});
			if (response && response.cursor) {
				this.eventsCursor = response.cursor;
			}
		} catch (e) {
			console.warn('Indexer: Failed to initialize events cursor:', e);
		}
	}
}
