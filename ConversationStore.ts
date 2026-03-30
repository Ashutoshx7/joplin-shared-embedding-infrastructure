/**
 * Conversation persistence using sql.js
 * Stores chat history so conversations survive plugin restarts
 * Each conversation has multiple turns (user + assistant messages)
 */

const initSqlJs = require('sql.js');

export interface ConversationTurn {
	id: number;
	conversationId: string;
	role: 'user' | 'assistant';
	content: string;
	citations: string; // JSON array of citation objects
	timestamp: number;
}

export interface Conversation {
	id: string;
	title: string;
	createdAt: number;
	updatedAt: number;
}

export class ConversationStore {
	private db: any = null;
	private dbPath: string;
	private flushTimer: ReturnType<typeof setInterval> | null = null;

	constructor(dbPath: string) {
		this.dbPath = dbPath;
	}

	public async initialize(): Promise<void> {
		const fs = require('fs');
		const SQL = await initSqlJs();

		let data: Buffer | null = null;
		try {
			data = fs.readFileSync(this.dbPath);
		} catch { /* new DB */ }

		this.db = new SQL.Database(data ? new Uint8Array(data) : undefined);
		this.createTables();

		// Auto-flush every 30 seconds
		this.flushTimer = setInterval(() => this.flush(), 30_000);
	}

	private createTables(): void {
		this.db.run(`
			CREATE TABLE IF NOT EXISTS conversations (
				id TEXT PRIMARY KEY,
				title TEXT NOT NULL,
				created_at INTEGER NOT NULL,
				updated_at INTEGER NOT NULL
			)
		`);

		this.db.run(`
			CREATE TABLE IF NOT EXISTS turns (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
				role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
				content TEXT NOT NULL,
				citations TEXT DEFAULT '[]',
				timestamp INTEGER NOT NULL
			)
		`);

		this.db.run(`CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conversation_id)`);
	}

	// ── Conversation CRUD ──

	public createConversation(title = 'New Chat'): string {
		const id = this.generateId();
		const now = Date.now();
		this.db.run(
			'INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)',
			[id, title, now, now]
		);
		return id;
	}

	public getConversations(): Conversation[] {
		const stmt = this.db.prepare('SELECT * FROM conversations ORDER BY updated_at DESC');
		const results: Conversation[] = [];
		while (stmt.step()) {
			const row = stmt.getAsObject();
			results.push({
				id: row.id as string,
				title: row.title as string,
				createdAt: row.created_at as number,
				updatedAt: row.updated_at as number,
			});
		}
		stmt.free();
		return results;
	}

	public deleteConversation(conversationId: string): void {
		this.db.run('DELETE FROM turns WHERE conversation_id = ?', [conversationId]);
		this.db.run('DELETE FROM conversations WHERE id = ?', [conversationId]);
	}

	public updateConversationTitle(conversationId: string, title: string): void {
		this.db.run('UPDATE conversations SET title = ? WHERE id = ?', [title, conversationId]);
	}

	// ── Turn CRUD ──

	public addTurn(conversationId: string, role: 'user' | 'assistant', content: string, citations: any[] = []): void {
		const now = Date.now();
		this.db.run(
			'INSERT INTO turns (conversation_id, role, content, citations, timestamp) VALUES (?, ?, ?, ?, ?)',
			[conversationId, role, content, JSON.stringify(citations), now]
		);
		this.db.run('UPDATE conversations SET updated_at = ? WHERE id = ?', [now, conversationId]);
	}

	public getTurns(conversationId: string): ConversationTurn[] {
		const stmt = this.db.prepare(
			'SELECT * FROM turns WHERE conversation_id = ? ORDER BY timestamp ASC',
			[conversationId]
		);
		const results: ConversationTurn[] = [];
		while (stmt.step()) {
			const row = stmt.getAsObject();
			results.push({
				id: row.id as number,
				conversationId: row.conversation_id as string,
				role: row.role as 'user' | 'assistant',
				content: row.content as string,
				citations: row.citations as string,
				timestamp: row.timestamp as number,
			});
		}
		stmt.free();
		return results;
	}

	/**
	 * Get turns as LLMMessage format for prompt building
	 */
	public getTurnsAsMessages(conversationId: string): { role: 'user' | 'assistant'; content: string }[] {
		return this.getTurns(conversationId).map(t => ({
			role: t.role,
			content: t.content,
		}));
	}

	// ── Persistence ──

	public flush(): void {
		if (!this.db) return;
		const fs = require('fs');
		const data = this.db.export();
		fs.writeFileSync(this.dbPath, Buffer.from(data));
	}

	public dispose(): void {
		if (this.flushTimer) clearInterval(this.flushTimer);
		this.flush();
		if (this.db) this.db.close();
	}

	// ── Helpers ──

	private generateId(): string {
		return `conv_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
	}

	public getStats(): { conversations: number; turns: number } {
		const convCount = this.db.exec('SELECT COUNT(*) FROM conversations')[0]?.values[0][0] || 0;
		const turnCount = this.db.exec('SELECT COUNT(*) FROM turns')[0]?.values[0][0] || 0;
		return { conversations: convCount as number, turns: turnCount as number };
	}
}
