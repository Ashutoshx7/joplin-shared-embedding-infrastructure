/**
 * Cross-encoder reranker
 *
 * Re-scores top-k results using a more precise relevance model.
 * Supports Ollama (via generate endpoint) and OpenAI-compatible APIs.
 *
 * @shikuz: "Reranking: using a second, more precise model to re-score
 *           the top results before passing them to the LLM."
 * @adamoutler: "On smaller models, reranking is important to ensure
 *              the model context is managed."
 */

import { ScoredChunk } from './vectorStore';

export interface RerankerConfig {
	enabled: boolean;
	provider: 'ollama' | 'openai' | 'none';
	endpoint?: string;
	model?: string;
	apiKey?: string;
	topK?: number;        // How many results to rerank (default: 20)
	returnK?: number;     // How many to return after reranking (default: 10)
}

export interface RerankResult {
	chunk: ScoredChunk;
	rerankScore: number;
}

/**
 * Reranks a list of candidate chunks by relevance to the query.
 *
 * Uses a cross-encoder approach: for each (query, chunk) pair,
 * asks the model to rate relevance on a 0-10 scale.
 * This is more precise than bi-encoder similarity but slower,
 * so we only apply it to the top-k candidates.
 */
export class Reranker {
	private config: RerankerConfig;

	constructor(config: RerankerConfig) {
		this.config = {
			topK: 20,
			returnK: 10,
			...config,
		};
	}

	get enabled(): boolean {
		return this.config.enabled && this.config.provider !== 'none';
	}

	/**
	 * Rerank candidates by relevance to the query.
	 * Returns the top returnK results sorted by rerank score.
	 */
	async rerank(query: string, candidates: ScoredChunk[]): Promise<ScoredChunk[]> {
		if (!this.enabled || candidates.length === 0) {
			return candidates;
		}

		// Only rerank the top-k candidates
		const toRerank = candidates.slice(0, this.config.topK!);
		const rest = candidates.slice(this.config.topK!);

		try {
			const scored = await this.scoreAll(query, toRerank);

			// Sort by rerank score descending
			scored.sort((a, b) => b.rerankScore - a.rerankScore);

			// Take top returnK and merge scores
			const reranked = scored.slice(0, this.config.returnK!).map(r => ({
				...r.chunk,
				score: r.rerankScore, // Replace with rerank score
			}));

			return reranked;
		} catch (e) {
			console.warn('Reranker failed, falling back to original order:', e);
			return candidates;
		}
	}

	private async scoreAll(query: string, chunks: ScoredChunk[]): Promise<RerankResult[]> {
		if (this.config.provider === 'ollama') {
			return this.scoreViaOllama(query, chunks);
		} else if (this.config.provider === 'openai') {
			return this.scoreViaOpenAI(query, chunks);
		}
		return chunks.map(c => ({ chunk: c, rerankScore: c.score }));
	}

	/**
	 * Score via Ollama's generate endpoint.
	 * Uses a lightweight prompt to get a relevance score.
	 */
	private async scoreViaOllama(query: string, chunks: ScoredChunk[]): Promise<RerankResult[]> {
		const endpoint = this.config.endpoint || 'http://localhost:11434';
		const model = this.config.model || 'llama3.2:1b';
		const results: RerankResult[] = [];

		// Process in parallel batches of 5 to limit concurrency
		const batchSize = 5;
		for (let i = 0; i < chunks.length; i += batchSize) {
			const batch = chunks.slice(i, i + batchSize);
			const promises = batch.map(async (chunk) => {
				const score = await this.scoreSingleOllama(endpoint, model, query, chunk.text);
				return { chunk, rerankScore: score };
			});
			results.push(...await Promise.all(promises));
		}

		return results;
	}

	private async scoreSingleOllama(
		endpoint: string,
		model: string,
		query: string,
		passage: string,
	): Promise<number> {
		const prompt = `Rate the relevance of the following passage to the query on a scale of 0 to 10. Only respond with a single number.

Query: ${query}
Passage: ${passage.substring(0, 500)}

Relevance score (0-10):`;

		try {
			const response = await fetch(`${endpoint}/api/generate`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model,
					prompt,
					stream: false,
					options: { temperature: 0, num_predict: 5 },
				}),
			});

			if (!response.ok) return 0;
			const data = await response.json();
			const text = (data.response || '').trim();
			const score = parseFloat(text);
			return isNaN(score) ? 0 : Math.min(Math.max(score / 10, 0), 1);
		} catch {
			return 0;
		}
	}

	/**
	 * Score via OpenAI-compatible chat endpoint.
	 */
	private async scoreViaOpenAI(query: string, chunks: ScoredChunk[]): Promise<RerankResult[]> {
		const endpoint = this.config.endpoint || 'https://api.openai.com/v1';
		const model = this.config.model || 'gpt-4o-mini';
		const results: RerankResult[] = [];

		for (const chunk of chunks) {
			try {
				const response = await fetch(`${endpoint}/chat/completions`, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						'Authorization': `Bearer ${this.config.apiKey}`,
					},
					body: JSON.stringify({
						model,
						messages: [
							{
								role: 'system',
								content: 'Rate the relevance of the passage to the query. Respond with ONLY a number from 0 to 10.',
							},
							{
								role: 'user',
								content: `Query: ${query}\nPassage: ${chunk.text.substring(0, 500)}`,
							},
						],
						temperature: 0,
						max_tokens: 5,
					}),
				});

				if (!response.ok) {
					results.push({ chunk, rerankScore: chunk.score });
					continue;
				}

				const data = await response.json();
				const text = data.choices?.[0]?.message?.content?.trim() || '0';
				const score = parseFloat(text);
				results.push({
					chunk,
					rerankScore: isNaN(score) ? 0 : Math.min(Math.max(score / 10, 0), 1),
				});
			} catch {
				results.push({ chunk, rerankScore: chunk.score });
			}
		}

		return results;
	}

	dispose(): void {}
}
