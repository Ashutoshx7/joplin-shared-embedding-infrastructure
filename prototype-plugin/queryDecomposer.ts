/**
 * Query decomposition
 *
 * Breaks complex multi-concept queries into sub-queries,
 * retrieves for each independently, then merges results via RRF.
 *
 * @shikuz: "Query decomposition: breaking complex questions into
 *           sub-queries and retrieving for each one separately."
 *
 * Uses simple heuristic decomposition (no LLM needed) plus
 * an optional LLM-assisted mode for complex natural language queries.
 */

export interface DecomposedQuery {
	original: string;
	subQueries: string[];
	strategy: 'single' | 'heuristic' | 'llm';
}

export interface QueryDecomposerConfig {
	enabled: boolean;
	/** Use LLM to decompose complex queries. Requires Ollama or OpenAI. */
	llmAssisted: boolean;
	provider?: 'ollama' | 'openai';
	endpoint?: string;
	model?: string;
	apiKey?: string;
}

/**
 * Decomposes complex queries into sub-queries for better retrieval.
 *
 * Strategy 1 (heuristic): Splits on conjunctions, commas, "and", "or",
 * "about X that also Y", "related to X and Y".
 *
 * Strategy 2 (LLM-assisted): Asks a small LLM to break down the query.
 */
export class QueryDecomposer {
	private config: QueryDecomposerConfig;

	constructor(config: QueryDecomposerConfig) {
		this.config = config;
	}

	get enabled(): boolean {
		return this.config.enabled;
	}

	/**
	 * Decompose a query into sub-queries.
	 * Returns { original, subQueries, strategy }.
	 * If the query is simple, subQueries will just be [original].
	 */
	async decompose(query: string): Promise<DecomposedQuery> {
		if (!this.enabled) {
			return { original: query, subQueries: [query], strategy: 'single' };
		}

		// Try heuristic decomposition first
		const heuristic = this.heuristicDecompose(query);
		if (heuristic.length > 1) {
			return { original: query, subQueries: heuristic, strategy: 'heuristic' };
		}

		// If LLM-assisted is enabled and query seems complex, use LLM
		if (this.config.llmAssisted && this.isComplexQuery(query)) {
			try {
				const llmResult = await this.llmDecompose(query);
				if (llmResult.length > 1) {
					return { original: query, subQueries: llmResult, strategy: 'llm' };
				}
			} catch (e) {
				console.warn('LLM decomposition failed, using original query:', e);
			}
		}

		return { original: query, subQueries: [query], strategy: 'single' };
	}

	/**
	 * Heuristic decomposition — splits on natural language conjunctions.
	 */
	private heuristicDecompose(query: string): string[] {
		const trimmed = query.trim();

		// Pattern: "X and Y" / "X, Y" / "X or Y"
		// But not if the conjunction is part of a phrase like "search and replace"
		const conjunctionPatterns = [
			// "notes about databases that also mention performance"
			/^(.+?)\s+(?:that\s+)?also\s+(?:mention|discuss|include|cover|talk about|reference)\s+(.+)$/i,
			// "related to X and Y" / "about X and Y"
			/^(?:related to|about|regarding|concerning)\s+(.+?)\s+(?:and|&)\s+(.+)$/i,
			// "X compared to Y" / "X versus Y"
			/^(.+?)\s+(?:compared to|versus|vs\.?)\s+(.+)$/i,
			// Simple "X and Y" where both parts are substantial (>3 words each)
			/^(.{15,}?)\s+(?:and|&)\s+(.{15,})$/i,
		];

		for (const pattern of conjunctionPatterns) {
			const match = trimmed.match(pattern);
			if (match) {
				const parts = match.slice(1).map(s => s.trim()).filter(s => s.length > 2);
				if (parts.length >= 2) {
					return parts;
				}
			}
		}

		// Comma-separated list of topics (at least 2 items, each >5 chars)
		const commaParts = trimmed.split(/\s*,\s+/).filter(s => s.length > 5);
		if (commaParts.length >= 2 && commaParts.length <= 5) {
			return commaParts;
		}

		return [trimmed];
	}

	/**
	 * Check if a query seems complex enough to warrant LLM decomposition.
	 */
	private isComplexQuery(query: string): boolean {
		const words = query.trim().split(/\s+/);
		return words.length >= 8; // Only decompose longer queries
	}

	/**
	 * LLM-assisted decomposition via Ollama or OpenAI.
	 */
	private async llmDecompose(query: string): Promise<string[]> {
		if (this.config.provider === 'ollama') {
			return this.llmDecomposeOllama(query);
		} else if (this.config.provider === 'openai') {
			return this.llmDecomposeOpenAI(query);
		}
		return [query];
	}

	private async llmDecomposeOllama(query: string): Promise<string[]> {
		const endpoint = this.config.endpoint || 'http://localhost:11434';
		const model = this.config.model || 'llama3.2:1b';

		const prompt = `Break the following search query into 2-4 simpler sub-queries that together cover the original intent. Return ONLY the sub-queries, one per line, no numbering or bullets.

Query: ${query}

Sub-queries:`;

		const response = await fetch(`${endpoint}/api/generate`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				model,
				prompt,
				stream: false,
				options: { temperature: 0, num_predict: 100 },
			}),
		});

		if (!response.ok) return [query];
		const data = await response.json();
		return this.parseLlmOutput(data.response || '', query);
	}

	private async llmDecomposeOpenAI(query: string): Promise<string[]> {
		const endpoint = this.config.endpoint || 'https://api.openai.com/v1';
		const model = this.config.model || 'gpt-4o-mini';

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
						content: 'Break search queries into simpler sub-queries. Return ONLY sub-queries, one per line.',
					},
					{ role: 'user', content: query },
				],
				temperature: 0,
				max_tokens: 100,
			}),
		});

		if (!response.ok) return [query];
		const data = await response.json();
		const text = data.choices?.[0]?.message?.content || '';
		return this.parseLlmOutput(text, query);
	}

	private parseLlmOutput(text: string, original: string): string[] {
		const lines = text
			.split('\n')
			.map(l => l.replace(/^[\d\-\*\.\)\s]+/, '').trim())
			.filter(l => l.length > 3 && l.length < 200);

		if (lines.length >= 2 && lines.length <= 5) {
			return lines;
		}
		return [original];
	}

	dispose(): void {}
}
