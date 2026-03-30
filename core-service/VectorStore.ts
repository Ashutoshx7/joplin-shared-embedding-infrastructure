/**
 * Vector store using sql.js (WASM SQLite) with custom cosine_sim()
 * In-memory DB with periodic disk flush.
 */

import type { NoteChunk } from './ChunkingEngine';

type SqlJsDatabase = any;
type SqlJsStatic = any;

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
	lastUpdated: number;
	dbSizeBytes: number;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS note_chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  note_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
  heading_path TEXT DEFAULT '', text TEXT NOT NULL,
  note_title TEXT NOT NULL, notebook_id TEXT DEFAULT '',
  content_hash TEXT NOT NULL, updated_time INTEGER NOT NULL,
  UNIQUE(note_id, chunk_index)
);
CREATE TABLE IF NOT EXISTS chunk_embeddings (
  chunk_id INTEGER PRIMARY KEY REFERENCES note_chunks(id) ON DELETE CASCADE,
  embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS note_embeddings (
  note_id TEXT PRIMARY KEY, embedding BLOB NOT NULL,
  chunk_count INTEGER NOT NULL, updated_time INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_chunks_note ON note_chunks(note_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON note_chunks(content_hash);
`;

export class VectorStore {
	private db: SqlJsDatabase = null;
	private SQL: SqlJsStatic = null;
	private dbPath: string;
	private flushInterval: any = null;
	private dirty = false;

	constructor(dbPath: string) { this.dbPath = dbPath; }

	async init(): Promise<void> {
		// eslint-disable-next-line no-eval -- dynamic require to avoid esbuild bundling
		const initSqlJs = eval('require')('sql.js');
		this.SQL = await initSqlJs();

		// Load existing DB
		let dbData: Uint8Array = null;
		try {
			// eslint-disable-next-line no-eval -- dynamic require to avoid esbuild bundling
			const fs = eval('require')('fs-extra');
			if (await fs.pathExists(this.dbPath)) {
				dbData = await fs.readFile(this.dbPath);
			}
		} catch (_e) { /* new DB */ }

		this.db = dbData ? new this.SQL.Database(dbData) : new this.SQL.Database();
		this.db.run(SCHEMA);

		// cosine_sim: dot product on pre-normalized vectors
		this.db.create_function('cosine_sim', (a: Uint8Array, b: Uint8Array) => {
			const fa = new Float32Array(a.buffer, a.byteOffset, a.byteLength / 4);
			const fb = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
			let dot = 0;
			for (let i = 0; i < fa.length; i++) dot += fa[i] * fb[i];
			return dot;
		});

		this.flushInterval = setInterval(() => { if (this.dirty) void this.flushToDisk(); }, 30_000);
	}

	storeNoteChunks(chunks: NoteChunk[], embeddings: Float32Array[]): void {
		if (!this.db || chunks.length === 0) return;
		const noteId = chunks[0].noteId;
		this.deleteNote(noteId);

		const insChunk = this.db.prepare(
			'INSERT INTO note_chunks (note_id,chunk_index,heading_path,text,note_title,notebook_id,content_hash,updated_time) VALUES (?,?,?,?,?,?,?,?)',
		);
		const insEmb = this.db.prepare('INSERT INTO chunk_embeddings (chunk_id,embedding) VALUES (?,?)');

		this.db.run('BEGIN');
		try {
			for (let i = 0; i < chunks.length; i++) {
				const c = chunks[i];
				insChunk.run([c.noteId, c.chunkIndex, c.headingPath, c.text, c.noteTitle, c.notebookId, c.contentHash, Date.now()]);
				const chunkId = this.db.exec('SELECT last_insert_rowid()')[0].values[0][0];
				const blob = new Uint8Array(embeddings[i].buffer, embeddings[i].byteOffset, embeddings[i].byteLength);
				insEmb.run([chunkId, blob]);
			}

			// Note-level embedding (mean)
			const noteEmb = this.meanEmbedding(embeddings);
			const noteBlob = new Uint8Array(noteEmb.buffer, noteEmb.byteOffset, noteEmb.byteLength);
			this.db.run('INSERT OR REPLACE INTO note_embeddings (note_id,embedding,chunk_count,updated_time) VALUES (?,?,?,?)',
				[noteId, noteBlob, chunks.length, Date.now()]);

			this.db.run('COMMIT');
		} catch (e) {
			this.db.run('ROLLBACK');
			throw e;
		} finally {
			insChunk.free();
			insEmb.free();
		}
		this.dirty = true;
	}

	searchByVector(queryEmb: Float32Array, limit = 50, notebookId?: string): ScoredChunk[] {
		if (!this.db) return [];
		const blob = new Uint8Array(queryEmb.buffer, queryEmb.byteOffset, queryEmb.byteLength);

		let sql = 'SELECT c.*, cosine_sim(ce.embedding, ?) AS score FROM chunk_embeddings ce JOIN note_chunks c ON c.id=ce.chunk_id';
		const params: any[] = [blob];

		if (notebookId) {
			sql += ' WHERE c.notebook_id = ?';
			params.push(notebookId);
		}

		sql += ' ORDER BY score DESC LIMIT ?';
		params.push(limit);

		const res = this.db.exec(sql, params);
		if (!res.length) return [];
		const cols = res[0].columns;
		return res[0].values.map((row: any[]) => {
			const o: any = {};
			cols.forEach((c: string, i: number) => o[c] = row[i]);
			return { id: o.id, noteId: o.note_id, chunkIndex: o.chunk_index, headingPath: o.heading_path, text: o.text, noteTitle: o.note_title, notebookId: o.notebook_id, contentHash: o.content_hash, updatedTime: o.updated_time, score: o.score } as ScoredChunk;
		});
	}

	findSimilarNotes(noteId: string, limit = 5): ScoredChunk[] {
		if (!this.db) return [];
		const r = this.db.exec('SELECT embedding FROM note_embeddings WHERE note_id=?', [noteId]);
		if (!r.length || !r[0].values.length) return [];
		const blob = r[0].values[0][0] as Uint8Array;
		const sr = this.db.exec(
			'SELECT ne.note_id, cosine_sim(ne.embedding, ?) AS score FROM note_embeddings ne WHERE ne.note_id!=? ORDER BY score DESC LIMIT ?',
			[blob, noteId, limit],
		);
		if (!sr.length) return [];
		return sr[0].values.map((row: any[]) => ({
			id: 0, noteId: row[0] as string, chunkIndex: 0, headingPath: '', text: '', noteTitle: '', notebookId: '', contentHash: '', updatedTime: 0, score: row[1] as number,
		}));
	}

	isNoteIndexed(noteId: string, hash: string): boolean {
		if (!this.db) return false;
		const r = this.db.exec('SELECT COUNT(*) FROM note_chunks WHERE note_id=? AND content_hash=?', [noteId, hash]);
		return r.length > 0 && r[0].values[0][0] > 0;
	}

	deleteNote(noteId: string): void {
		if (!this.db) return;
		this.db.run('DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT id FROM note_chunks WHERE note_id=?)', [noteId]);
		this.db.run('DELETE FROM note_chunks WHERE note_id=?', [noteId]);
		this.db.run('DELETE FROM note_embeddings WHERE note_id=?', [noteId]);
		this.dirty = true;
	}

	getStats(): IndexStats {
		if (!this.db) return { totalNotes: 0, totalChunks: 0, modelName: '', lastUpdated: 0, dbSizeBytes: 0 };
		const nc = this.db.exec('SELECT COUNT(DISTINCT note_id) FROM note_chunks');
		const cc = this.db.exec('SELECT COUNT(*) FROM note_chunks');
		const mm = this.db.exec("SELECT value FROM index_meta WHERE key='model_name'");
		const lu = this.db.exec('SELECT MAX(updated_time) FROM note_chunks');
		const exp = this.db.export();
		return {
			totalNotes: nc.length ? nc[0].values[0][0] : 0,
			totalChunks: cc.length ? cc[0].values[0][0] : 0,
			modelName: mm.length && mm[0].values.length ? mm[0].values[0][0] : 'unknown',
			lastUpdated: lu.length && lu[0].values.length ? lu[0].values[0][0] : 0,
			dbSizeBytes: exp.byteLength,
		};
	}

	setMeta(key: string, value: string): void {
		if (!this.db) return;
		this.db.run('INSERT OR REPLACE INTO index_meta (key,value) VALUES (?,?)', [key, value]);
		this.dirty = true;
	}

	getNoteEmbedding(noteId: string): number[] | null {
		if (!this.db) return null;
		const r = this.db.exec('SELECT embedding FROM note_embeddings WHERE note_id=?', [noteId]);
		if (!r.length || !r[0].values.length) return null;
		const blob = r[0].values[0][0] as Uint8Array;
		const vec = new Float32Array(blob.buffer, blob.byteOffset, blob.byteLength / 4);
		return Array.from(vec);
	}

	getAllNoteEmbeddings(): { noteId: string; embedding: number[] }[] {
		if (!this.db) return [];
		const r = this.db.exec('SELECT note_id, embedding FROM note_embeddings');
		if (!r.length) return [];
		return r[0].values.map((row: any[]) => {
			const blob = row[1] as Uint8Array;
			const vec = new Float32Array(blob.buffer, blob.byteOffset, blob.byteLength / 4);
			return { noteId: row[0] as string, embedding: Array.from(vec) };
		});
	}

	getChunkEmbeddings(noteId: string): number[][] {
		if (!this.db) return [];
		const r = this.db.exec(
			'SELECT ce.embedding FROM chunk_embeddings ce JOIN note_chunks c ON c.id=ce.chunk_id WHERE c.note_id=? ORDER BY c.chunk_index',
			[noteId],
		);
		if (!r.length) return [];
		return r[0].values.map((row: any[]) => {
			const blob = row[0] as Uint8Array;
			const vec = new Float32Array(blob.buffer, blob.byteOffset, blob.byteLength / 4);
			return Array.from(vec);
		});
	}

	clear(): void {
		if (!this.db) return;
		this.db.run('DELETE FROM chunk_embeddings');
		this.db.run('DELETE FROM note_chunks');
		this.db.run('DELETE FROM note_embeddings');
		this.db.run('DELETE FROM index_meta');
		this.dirty = true;
	}

	async flushToDisk(): Promise<void> {
		if (!this.db) return;
		try {
			// eslint-disable-next-line no-eval -- dynamic require to avoid esbuild bundling
			const fs = eval('require')('fs-extra');
			const data = this.db.export();
			await fs.writeFile(this.dbPath, Buffer.from(data));
			this.dirty = false;
		} catch (e) {
			console.error('VectorStore flush failed:', e);
		}
	}

	async close(): Promise<void> {
		if (this.flushInterval) { clearInterval(this.flushInterval); this.flushInterval = null; }
		await this.flushToDisk();
		if (this.db) { this.db.close(); this.db = null; }
	}

	private meanEmbedding(embeddings: Float32Array[]): Float32Array {
		if (!embeddings.length) return new Float32Array(0);
		const dim = embeddings[0].length;
		const avg = new Float32Array(dim);
		for (const e of embeddings) for (let i = 0; i < dim; i++) avg[i] += e[i];
		let norm = 0;
		for (let i = 0; i < dim; i++) norm += avg[i] * avg[i];
		norm = Math.sqrt(norm);
		if (norm > 0) for (let i = 0; i < dim; i++) avg[i] /= norm;
		return avg;
	}
}
