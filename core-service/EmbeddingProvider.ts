/**
 * Embedding provider abstraction — Local / Ollama / OpenAI
 */

export interface EmbeddingProvider {
	readonly name: string;
	readonly dimensions: number;
	embed(text: string): Promise<Float32Array>;
	embedBatch(texts: string[]): Promise<Float32Array[]>;
	embedForQuery(query: string): Promise<Float32Array>;
	dispose(): void;
}

export type ProviderType = 'local' | 'ollama' | 'openai';

// ─── Ollama HTTP Provider (default — no extra deps) ─────────────────────────

export class OllamaEmbeddingProvider implements EmbeddingProvider {
	readonly name = 'ollama';
	readonly dimensions: number;
	private endpoint: string;
	private model: string;

	constructor(endpoint = 'http://localhost:11434', model = 'nomic-embed-text', dimensions = 768) {
		this.endpoint = endpoint;
		this.model = model;
		this.dimensions = dimensions;
	}

	async embed(text: string): Promise<Float32Array> {
		const response = await fetch(`${this.endpoint}/api/embeddings`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ model: this.model, prompt: text }),
		});
		if (!response.ok) throw new Error(`Ollama: ${response.statusText}`);
		const data: any = await response.json();
		const emb = new Float32Array(data.embedding);
		// Normalize
		let norm = 0;
		for (let i = 0; i < emb.length; i++) norm += emb[i] * emb[i];
		norm = Math.sqrt(norm);
		if (norm > 0) for (let i = 0; i < emb.length; i++) emb[i] /= norm;
		return emb;
	}

	async embedBatch(texts: string[]): Promise<Float32Array[]> {
		return Promise.all(texts.map(t => this.embed(t)));
	}

	async embedForQuery(query: string): Promise<Float32Array> {
		return this.embed(`search_query: ${query}`);
	}

	dispose(): void {}
}

// ─── OpenAI-compatible Provider ─────────────────────────────────────────────

export class OpenAIEmbeddingProvider implements EmbeddingProvider {
	readonly name = 'openai';
	readonly dimensions: number;
	private endpoint: string;
	private apiKey: string;
	private model: string;

	constructor(apiKey: string, model = 'text-embedding-3-small', dimensions = 1536, endpoint = 'https://api.openai.com/v1') {
		this.apiKey = apiKey;
		this.model = model;
		this.dimensions = dimensions;
		this.endpoint = endpoint;
	}

	async embed(text: string): Promise<Float32Array> {
		const response = await fetch(`${this.endpoint}/embeddings`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${this.apiKey}` },
			body: JSON.stringify({ model: this.model, input: text }),
		});
		if (!response.ok) throw new Error(`OpenAI: ${response.statusText}`);
		const data: any = await response.json();
		const emb = new Float32Array(data.data[0].embedding);
		let norm = 0;
		for (let i = 0; i < emb.length; i++) norm += emb[i] * emb[i];
		norm = Math.sqrt(norm);
		if (norm > 0) for (let i = 0; i < emb.length; i++) emb[i] /= norm;
		return emb;
	}

	async embedBatch(texts: string[]): Promise<Float32Array[]> {
		const response = await fetch(`${this.endpoint}/embeddings`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${this.apiKey}` },
			body: JSON.stringify({ model: this.model, input: texts }),
		});
		if (!response.ok) throw new Error(`OpenAI batch: ${response.statusText}`);
		const data: any = await response.json();
		return data.data.map((d: any) => {
			const emb = new Float32Array(d.embedding);
			let norm = 0;
			for (let i = 0; i < emb.length; i++) norm += emb[i] * emb[i];
			norm = Math.sqrt(norm);
			if (norm > 0) for (let i = 0; i < emb.length; i++) emb[i] /= norm;
			return emb;
		});
	}

	async embedForQuery(query: string): Promise<Float32Array> {
		return this.embed(query);
	}

	dispose(): void {}
}

// ─── Factory ────────────────────────────────────────────────────────────────

export function createEmbeddingProvider(
	type: ProviderType,
	config: { endpoint?: string; modelId?: string; apiKey?: string; dimensions?: number; cacheDir?: string } = {},
): EmbeddingProvider {
	switch (type) {
		case 'ollama':
			return new OllamaEmbeddingProvider(config.endpoint, config.modelId, config.dimensions);
		case 'openai':
			return new OpenAIEmbeddingProvider(config.apiKey || '', config.modelId, config.dimensions, config.endpoint);
		case 'local':
			// Local WASM provider — requires @xenova/transformers (GSoC deliverable)
			// For now, fallback to Ollama
			return new OllamaEmbeddingProvider(config.endpoint, config.modelId, config.dimensions);
		default:
			throw new Error(`Unknown provider: ${type}`);
	}
}
