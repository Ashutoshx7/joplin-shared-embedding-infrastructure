/**
 * Embedding engine abstraction
 *
 * Provides a provider-agnostic interface for generating embeddings.
 * Default: local inference via Transformers.js (WASM) in a Web Worker.
 * Fallback: Ollama HTTP or OpenAI-compatible API.
 */

export interface EmbeddingProvider {
	readonly name: string;
	readonly dimensions: number;
	embed(text: string): Promise<Float32Array>;
	embedBatch(texts: string[]): Promise<Float32Array[]>;
	embedForQuery(query: string): Promise<Float32Array>;
	dispose(): void;
}

// ─── Local WASM Provider (Transformers.js) ──────────────────────────────────

/**
 * Local embedding using Transformers.js with BGE-small-en-v1.5.
 * All inference runs in-process via WASM. No API key needed.
 *
 * NOTE: In production, this would run inside a Web Worker to avoid
 * blocking the plugin's event loop. For the prototype, we run in-process
 * and will add Worker support as a follow-up.
 */
export class LocalEmbeddingProvider implements EmbeddingProvider {
	readonly name = 'local-bge-small';
	readonly dimensions = 384;

	private pipeline: any = null;
	private callCount = 0;
	private modelId: string;
	private cacheDir: string;

	constructor(modelId: string = 'Xenova/bge-small-en-v1.5', cacheDir: string = '') {
		this.modelId = modelId;
		this.cacheDir = cacheDir;
	}

	async init(): Promise<void> {
		try {
			const transformers = require('@xenova/transformers');
			if (this.cacheDir) {
				transformers.env.cacheDir = this.cacheDir;
			}
			transformers.env.allowRemoteModels = true;
			this.pipeline = await transformers.pipeline('feature-extraction', this.modelId);
		} catch (e) {
			throw new Error(
				'Local embedding provider requires @xenova/transformers. ' +
				'Install it with: npm install @xenova/transformers, ' +
				'or switch to Ollama/OpenAI provider in settings. Error: ' + e,
			);
		}
	}

	private async ensurePipeline(): Promise<void> {
		if (!this.pipeline) {
			await this.init();
		}
		// Recycle WASM pipeline every 80 calls to prevent memory fragmentation
		if (this.callCount > 0 && this.callCount % 80 === 0) {
			this.pipeline?.dispose?.();
			const transformers = require('@xenova/transformers');
			this.pipeline = await transformers.pipeline('feature-extraction', this.modelId);
		}
	}

	async embed(text: string): Promise<Float32Array> {
		await this.ensurePipeline();
		const output = await this.pipeline(text, { pooling: 'mean', normalize: true });
		this.callCount++;
		return new Float32Array(output.data);
	}

	async embedBatch(texts: string[]): Promise<Float32Array[]> {
		const results: Float32Array[] = [];
		for (const text of texts) {
			results.push(await this.embed(text));
		}
		return results;
	}

	async embedForQuery(query: string): Promise<Float32Array> {
		// BGE-small: add retrieval prefix for queries per model card
		return this.embed(`Represent this sentence for searching relevant passages: ${query}`);
	}

	dispose(): void {
		this.pipeline?.dispose?.();
		this.pipeline = null;
	}
}

// ─── Ollama HTTP Provider ───────────────────────────────────────────────────

export class OllamaEmbeddingProvider implements EmbeddingProvider {
	readonly name = 'ollama';
	readonly dimensions: number;

	private endpoint: string;
	private model: string;

	constructor(endpoint: string = 'http://localhost:11434', model: string = 'nomic-embed-text', dimensions: number = 768) {
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

		if (!response.ok) {
			throw new Error(`Ollama embedding failed: ${response.statusText}`);
		}

		const data = await response.json();
		const embedding = new Float32Array(data.embedding);
		// Normalize to unit vector
		let norm = 0;
		for (let i = 0; i < embedding.length; i++) norm += embedding[i] * embedding[i];
		norm = Math.sqrt(norm);
		if (norm > 0) {
			for (let i = 0; i < embedding.length; i++) embedding[i] /= norm;
		}
		return embedding;
	}

	async embedBatch(texts: string[]): Promise<Float32Array[]> {
		return Promise.all(texts.map(t => this.embed(t)));
	}

	async embedForQuery(query: string): Promise<Float32Array> {
		return this.embed(`search_query: ${query}`);
	}

	dispose(): void {}
}

// ─── OpenAI-compatible HTTP Provider ────────────────────────────────────────

export class OpenAIEmbeddingProvider implements EmbeddingProvider {
	readonly name = 'openai';
	readonly dimensions: number;

	private endpoint: string;
	private apiKey: string;
	private model: string;

	constructor(
		apiKey: string,
		model: string = 'text-embedding-3-small',
		dimensions: number = 1536,
		endpoint: string = 'https://api.openai.com/v1',
	) {
		this.apiKey = apiKey;
		this.model = model;
		this.dimensions = dimensions;
		this.endpoint = endpoint;
	}

	async embed(text: string): Promise<Float32Array> {
		const response = await fetch(`${this.endpoint}/embeddings`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Authorization': `Bearer ${this.apiKey}`,
			},
			body: JSON.stringify({ model: this.model, input: text }),
		});

		if (!response.ok) {
			throw new Error(`OpenAI embedding failed: ${response.statusText}`);
		}

		const data = await response.json();
		const embedding = new Float32Array(data.data[0].embedding);
		// Normalize
		let norm = 0;
		for (let i = 0; i < embedding.length; i++) norm += embedding[i] * embedding[i];
		norm = Math.sqrt(norm);
		if (norm > 0) {
			for (let i = 0; i < embedding.length; i++) embedding[i] /= norm;
		}
		return embedding;
	}

	async embedBatch(texts: string[]): Promise<Float32Array[]> {
		// OpenAI supports batch in a single call
		const response = await fetch(`${this.endpoint}/embeddings`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Authorization': `Bearer ${this.apiKey}`,
			},
			body: JSON.stringify({ model: this.model, input: texts }),
		});

		if (!response.ok) {
			throw new Error(`OpenAI batch embedding failed: ${response.statusText}`);
		}

		const data = await response.json();
		return data.data.map((d: any) => {
			const emb = new Float32Array(d.embedding);
			let norm = 0;
			for (let i = 0; i < emb.length; i++) norm += emb[i] * emb[i];
			norm = Math.sqrt(norm);
			if (norm > 0) {
				for (let i = 0; i < emb.length; i++) emb[i] /= norm;
			}
			return emb;
		});
	}

	async embedForQuery(query: string): Promise<Float32Array> {
		return this.embed(query);
	}

	dispose(): void {}
}

// ─── Factory ────────────────────────────────────────────────────────────────

export type ProviderType = 'local' | 'ollama' | 'openai';

export function createEmbeddingProvider(
	type: ProviderType,
	config: {
		cacheDir?: string;
		modelId?: string;
		endpoint?: string;
		apiKey?: string;
		dimensions?: number;
	} = {},
): EmbeddingProvider {
	switch (type) {
		case 'local':
			return new LocalEmbeddingProvider(config.modelId, config.cacheDir);
		case 'ollama':
			return new OllamaEmbeddingProvider(config.endpoint, config.modelId, config.dimensions);
		case 'openai':
			return new OpenAIEmbeddingProvider(
				config.apiKey || '',
				config.modelId,
				config.dimensions,
				config.endpoint,
			);
		default:
			throw new Error(`Unknown provider type: ${type}`);
	}
}
