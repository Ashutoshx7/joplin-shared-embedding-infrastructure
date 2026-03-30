/**
 * sql.js-backed vector store
 *
 * Stores chunk embeddings in a SQLite database (via sql.js WASM).
 * Cosine similarity via dot product on pre-normalized vectors.
 * All vectors stored as binary BLOBs for compact storage.
 *
 * No native dependencies — sql.js runs entirely in WASM.
 */

import type { NoteChunk } from './chunker';

export interface StoredChunk {
	id: number;
	noteId: string;
	chunkIndex: number;
	headingPath: string;
	text: string;
	noteTitle: string;
	notebookId: string;
	contentHash: string;
	updatedTime: number;
}

export interface ScoredChunk extends StoredChunk {
	score: number;
}

export interface IndexStats {
	totalNotes: number;
	totalChunks: number;
	modelName: string;
	dimensions: number;
	lastUpdated: number;
	dbSizeBytes: number;
}

export class VectorStore {
	private db: any = null;
	private dbPath: string;
	private dirty = false;
	private flushInterval: ReturnType<typeof setInterval> | null = null;

	constructor(dbPath: string) {
		this.dbPath = dbPath;
	}

	async init(_installDir?: string): Promise<void> {
		const initSqlJs = require('sql-asm');
		const fs = require('fs-extra');

		let buffer: Buffer | null = null;
		try {
			if (fs && await fs.pathExists(this.dbPath)) {
				buffer = await fs.readFile(this.dbPath);
				console.info('AI Search: loading existing index from disk');
			}
		} catch (e) {
			console.info('AI Search: no existing index, starting fresh');
		}

		const SQL = await initSqlJs();
		this.db = buffer ? new SQL.Database(buffer) : new SQL.Database();

		this.createTables();

		const stats = this.getStats();
		console.info(`AI Search: initialized with ${stats.totalChunks} chunks, ${stats.totalNotes} notes`);

		// Flush to disk every 30 seconds if dirty
		this.flushInterval = setInterval(() => {
			if (this.dirty) void this.flushToDisk();
		}, 30_000);
	}

	private createTables(): void {
		this.db.run(`
			CREATE TABLE IF NOT EXISTS chunks (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				note_id TEXT NOT NULL,
				chunk_index INTEGER NOT NULL,
				heading_path TEXT DEFAULT '',
				text TEXT NOT NULL,
				note_title TEXT DEFAULT '',
				notebook_id TEXT DEFAULT '',
				content_hash TEXT NOT NULL,
				updated_time INTEGER NOT NULL,
				embedding BLOB NOT NULL
			)
		`);

		this.db.run(`
			CREATE INDEX IF NOT EXISTS idx_chunks_note_id ON chunks(note_id)
		`);

		this.db.run(`
			CREATE INDEX IF NOT EXISTS idx_chunks_notebook_id ON chunks(notebook_id)
		`);

		this.db.run(`
			CREATE TABLE IF NOT EXISTS note_embeddings (
				note_id TEXT PRIMARY KEY,
				embedding BLOB NOT NULL,
				chunk_count INTEGER NOT NULL,
				updated_time INTEGER NOT NULL
			)
		`);

		this.db.run(`
			CREATE TABLE IF NOT EXISTS meta (
				key TEXT PRIMARY KEY,
				value TEXT NOT NULL
			)
		`);

		// Register cosine_sim SQL custom function for SQL-level similarity queries
		// Pre-normalized vectors → dot product = cosine similarity
		this.db.create_function('cosine_sim', (aBlob: Uint8Array, bBlob: Uint8Array) => {
			const a = new Float32Array(aBlob.buffer, aBlob.byteOffset, aBlob.byteLength / 4);
			const b = new Float32Array(bBlob.buffer, bBlob.byteOffset, bBlob.byteLength / 4);
			let dot = 0;
			for (let i = 0; i < a.length && i < b.length; i++) dot += a[i] * b[i];
			return dot;
		});
	}

	// ─── Encode/Decode Vectors ────────────────────────────────────────

	private encodeVector(vec: Float32Array): Uint8Array {
		return new Uint8Array(vec.buffer, vec.byteOffset, vec.byteLength);
	}

	private decodeVector(blob: Uint8Array): Float32Array {
		const buffer = new ArrayBuffer(blob.length);
		new Uint8Array(buffer).set(blob);
		return new Float32Array(buffer);
	}

	// ─── Store Operations ─────────────────────────────────────────────

	storeNoteChunks(chunks: NoteChunk[], embeddings: Float32Array[]): void {
		if (chunks.length === 0) return;

		const noteId = chunks[0].noteId;
		this.deleteNote(noteId);

		const insertChunk = this.db.prepare(`
			INSERT INTO chunks (note_id, chunk_index, heading_path, text, note_title, notebook_id, content_hash, updated_time, embedding)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		`);

		const now = Date.now();

		for (let i = 0; i < chunks.length; i++) {
			const c = chunks[i];
			insertChunk.run([
				c.noteId, c.chunkIndex, c.headingPath, c.text,
				c.noteTitle, c.notebookId, c.contentHash, now,
				this.encodeVector(embeddings[i]),
			]);
		}
		insertChunk.free();

		// Note-level embedding (mean of chunk vectors, re-normalized)
		const noteEmb = this.meanEmbedding(embeddings);
		this.db.run(
			`INSERT OR REPLACE INTO note_embeddings (note_id, embedding, chunk_count, updated_time)
			 VALUES (?, ?, ?, ?)`,
			[noteId, this.encodeVector(noteEmb), chunks.length, now],
		);

		this.dirty = true;
	}

	// ─── Search Operations ────────────────────────────────────────────

	searchByVector(queryEmb: Float32Array, limit = 50, notebookId?: string): ScoredChunk[] {
		let query = 'SELECT id, note_id, chunk_index, heading_path, text, note_title, notebook_id, content_hash, updated_time, embedding FROM chunks';
		const params: any[] = [];

		if (notebookId) {
			query += ' WHERE notebook_id = ?';
			params.push(notebookId);
		}

		const stmt = this.db.prepare(query);
		if (params.length > 0) stmt.bind(params);

		const results: ScoredChunk[] = [];

		while (stmt.step()) {
			const row = stmt.getAsObject({ ':blob': true });
			const embedding = this.decodeVector(row.embedding as Uint8Array);

			let dot = 0;
			for (let i = 0; i < queryEmb.length && i < embedding.length; i++) {
				dot += queryEmb[i] * embedding[i];
			}

			results.push({
				id: row.id as number,
				noteId: row.note_id as string,
				chunkIndex: row.chunk_index as number,
				headingPath: row.heading_path as string,
				text: row.text as string,
				noteTitle: row.note_title as string,
				notebookId: row.notebook_id as string,
				contentHash: row.content_hash as string,
				updatedTime: row.updated_time as number,
				score: dot,
			});
		}
		stmt.free();

		results.sort((a, b) => b.score - a.score);
		return results.slice(0, limit);
	}

	findSimilarNotes(noteId: string, limit = 5): ScoredChunk[] {
		// Get target note's embedding
		const targetStmt = this.db.prepare('SELECT embedding FROM note_embeddings WHERE note_id = ?');
		targetStmt.bind([noteId]);
		if (!targetStmt.step()) {
			targetStmt.free();
			return [];
		}
		const targetRow = targetStmt.getAsObject({ ':blob': true });
		const targetEmb = this.decodeVector(targetRow.embedding as Uint8Array);
		targetStmt.free();

		// Compare against all other notes
		const stmt = this.db.prepare('SELECT note_id, embedding FROM note_embeddings WHERE note_id != ?');
		stmt.bind([noteId]);

		const results: ScoredChunk[] = [];

		while (stmt.step()) {
			const row = stmt.getAsObject({ ':blob': true });
			const emb = this.decodeVector(row.embedding as Uint8Array);

			let dot = 0;
			for (let i = 0; i < targetEmb.length && i < emb.length; i++) {
				dot += targetEmb[i] * emb[i];
			}

			results.push({
				id: 0, noteId: row.note_id as string, chunkIndex: 0,
				headingPath: '', text: '', noteTitle: '', notebookId: '',
				contentHash: '', updatedTime: 0, score: dot,
			});
		}
		stmt.free();

		results.sort((a, b) => b.score - a.score);
		return results.slice(0, limit);
	}

	// ─── Note Embedding Access ────────────────────────────────────────

	getNoteEmbedding(noteId: string): Float32Array | null {
		const stmt = this.db.prepare('SELECT embedding FROM note_embeddings WHERE note_id = ?');
		stmt.bind([noteId]);
		if (!stmt.step()) {
			stmt.free();
			return null;
		}
		const row = stmt.getAsObject({ ':blob': true });
		stmt.free();
		return this.decodeVector(row.embedding as Uint8Array);
	}

	getAllNoteEmbeddings(): { noteId: string; embedding: Float32Array }[] {
		const stmt = this.db.prepare('SELECT note_id, embedding FROM note_embeddings');
		const results: { noteId: string; embedding: Float32Array }[] = [];

		while (stmt.step()) {
			const row = stmt.getAsObject({ ':blob': true });
			results.push({
				noteId: row.note_id as string,
				embedding: this.decodeVector(row.embedding as Uint8Array),
			});
		}
		stmt.free();
		return results;
	}

	getChunkEmbeddings(noteId: string): Float32Array[] {
		const stmt = this.db.prepare(
			'SELECT embedding FROM chunks WHERE note_id = ? ORDER BY chunk_index',
		);
		stmt.bind([noteId]);
		const results: Float32Array[] = [];
		while (stmt.step()) {
			const row = stmt.getAsObject({ ':blob': true });
			results.push(this.decodeVector(row.embedding as Uint8Array));
		}
		stmt.free();
		return results;
	}

	// ─── Change Detection ─────────────────────────────────────────────

	isNoteIndexed(noteId: string, hash: string): boolean {
		const stmt = this.db.prepare(
			'SELECT COUNT(*) as cnt FROM chunks WHERE note_id = ? AND content_hash = ?',
		);
		stmt.bind([noteId, hash]);
		stmt.step();
		const row = stmt.getAsObject();
		stmt.free();
		return (row.cnt as number) > 0;
	}

	getIndexedNoteIds(): Set<string> {
		const stmt = this.db.prepare('SELECT DISTINCT note_id FROM note_embeddings');
		const ids = new Set<string>();
		while (stmt.step()) {
			const row = stmt.getAsObject();
			ids.add(row.note_id as string);
		}
		stmt.free();
		return ids;
	}

	// ─── Delete / Clear ───────────────────────────────────────────────

	deleteNote(noteId: string): void {
		this.db.run('DELETE FROM chunks WHERE note_id = ?', [noteId]);
		this.db.run('DELETE FROM note_embeddings WHERE note_id = ?', [noteId]);
		this.dirty = true;
	}

	clear(): void {
		this.db.run('DELETE FROM chunks');
		this.db.run('DELETE FROM note_embeddings');
		this.db.run('DELETE FROM meta');
		this.dirty = true;
	}

	// ─── Meta ─────────────────────────────────────────────────────────

	setMeta(key: string, value: string): void {
		this.db.run(
			'INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)',
			[key, value],
		);
		this.dirty = true;
	}

	getMeta(key: string): string | null {
		const stmt = this.db.prepare('SELECT value FROM meta WHERE key = ?');
		stmt.bind([key]);
		if (!stmt.step()) {
			stmt.free();
			return null;
		}
		const row = stmt.getAsObject();
		stmt.free();
		return row.value as string;
	}

	// ─── Stats ────────────────────────────────────────────────────────

	getStats(): IndexStats {
		const chunkCount = this.scalarQuery('SELECT COUNT(*) as cnt FROM chunks');
		const noteCount = this.scalarQuery('SELECT COUNT(*) as cnt FROM note_embeddings');
		const lastUpdated = this.scalarQuery('SELECT MAX(updated_time) as cnt FROM chunks') || 0;

		return {
			totalNotes: noteCount,
			totalChunks: chunkCount,
			modelName: this.getMeta('model_name') || 'unknown',
			dimensions: parseInt(this.getMeta('dimensions') || '0', 10),
			lastUpdated,
			dbSizeBytes: this.db ? this.db.export().length : 0,
		};
	}

	/**
	 * Get note count without full stats (for cost estimation before indexing).
	 */
	getIndexedChunkCount(): number {
		return this.scalarQuery('SELECT COUNT(*) as cnt FROM chunks');
	}

	private scalarQuery(sql: string): number {
		try {
			const stmt = this.db.prepare(sql);
			stmt.step();
			const row = stmt.getAsObject();
			stmt.free();
			return (row.cnt as number) || 0;
		} catch {
			return 0;
		}
	}

	// ─── Persistence ──────────────────────────────────────────────────

	async flushToDisk(): Promise<void> {
		try {
			const fs = require('fs-extra');
			if (!fs || !this.db) return;

			const data = this.db.export();
			const buffer = Buffer.from(data);
			await fs.writeFile(this.dbPath, buffer);
			this.dirty = false;

			const stats = this.getStats();
			console.info(`AI Search: flushed ${stats.totalChunks} chunks to disk (${(buffer.length / 1024).toFixed(1)} KB)`);
		} catch (e) {
			console.error('AI Search: flush failed:', e);
		}
	}

	async close(): Promise<void> {
		if (this.flushInterval) {
			clearInterval(this.flushInterval);
			this.flushInterval = null;
		}
		await this.flushToDisk();
		if (this.db) {
			this.db.close();
			this.db = null;
		}
	}

	// ─── Internal Helpers ─────────────────────────────────────────────

	private meanEmbedding(embeddings: Float32Array[]): Float32Array {
		if (!embeddings.length) return new Float32Array(0);
		const dim = embeddings[0].length;
		const avg = new Float32Array(dim);
		for (const e of embeddings) {
			for (let i = 0; i < dim; i++) avg[i] += e[i];
		}
		let norm = 0;
		for (let i = 0; i < dim; i++) norm += avg[i] * avg[i];
		norm = Math.sqrt(norm);
		if (norm > 0) {
			for (let i = 0; i < dim; i++) avg[i] /= norm;
		}
		return avg;
	}
}
