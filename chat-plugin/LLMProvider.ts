/**
 * LLM Provider abstraction for Chat plugin
 * Supports OpenAI, Gemini (free tier), and Ollama (local)
 * All providers use streaming for real-time response display
 */

export interface LLMMessage {
	role: 'system' | 'user' | 'assistant';
	content: string;
}

export interface LLMStreamCallbacks {
	onToken: (token: string) => void;
	onDone: (fullText: string) => void;
	onError: (error: Error) => void;
}

export interface LLMProvider {
	readonly name: string;
	chat(messages: LLMMessage[], callbacks: LLMStreamCallbacks): Promise<void>;
	chatSync(messages: LLMMessage[]): Promise<string>;
	dispose(): void;
}

// ── OpenAI-Compatible Provider (works with OpenAI, Gemini, any compatible API) ──

export class OpenAIProvider implements LLMProvider {
	readonly name: string;
	private apiKey: string;
	private model: string;
	private baseUrl: string;

	constructor(apiKey: string, model = 'gpt-4o-mini', baseUrl = 'https://api.openai.com/v1', name = 'OpenAI') {
		this.apiKey = apiKey;
		this.model = model;
		this.baseUrl = baseUrl;
		this.name = name;
	}

	async chat(messages: LLMMessage[], callbacks: LLMStreamCallbacks): Promise<void> {
		try {
			const response = await fetch(`${this.baseUrl}/chat/completions`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Authorization': `Bearer ${this.apiKey}`,
				},
				body: JSON.stringify({
					model: this.model,
					messages,
					stream: true,
					temperature: 0.3,
					max_tokens: 2048,
				}),
			});

			if (!response.ok) {
				const errorBody = await response.text();
				throw new Error(`${this.name} API error ${response.status}: ${errorBody.substring(0, 200)}`);
			}

			const reader = response.body?.getReader();
			if (!reader) throw new Error('No response body');

			const decoder = new TextDecoder();
			let fullText = '';
			let buffer = '';

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n');
				buffer = lines.pop() || '';

				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;
					const data = line.slice(6).trim();
					if (data === '[DONE]') continue;

					try {
						const parsed = JSON.parse(data);
						const token = parsed.choices?.[0]?.delta?.content;
						if (token) {
							fullText += token;
							callbacks.onToken(token);
						}
					} catch { /* skip malformed SSE */ }
				}
			}

			callbacks.onDone(fullText);
		} catch (error) {
			callbacks.onError(error instanceof Error ? error : new Error(String(error)));
		}
	}

	async chatSync(messages: LLMMessage[]): Promise<string> {
		const response = await fetch(`${this.baseUrl}/chat/completions`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Authorization': `Bearer ${this.apiKey}`,
			},
			body: JSON.stringify({
				model: this.model,
				messages,
				temperature: 0.3,
				max_tokens: 2048,
			}),
		});

		if (!response.ok) throw new Error(`${this.name} API error ${response.status}`);
		const data = await response.json();
		return data.choices[0].message.content;
	}

	dispose() {}
}

// ── Gemini Provider (free tier via Google AI Studio) ──

export class GeminiProvider implements LLMProvider {
	readonly name = 'Gemini';
	private apiKey: string;
	private model: string;

	constructor(apiKey: string, model = 'gemini-2.0-flash') {
		this.apiKey = apiKey;
		this.model = model;
	}

	async chat(messages: LLMMessage[], callbacks: LLMStreamCallbacks): Promise<void> {
		try {
			// Convert OpenAI format to Gemini format
			const systemInstruction = messages.find(m => m.role === 'system')?.content || '';
			const contents = messages
				.filter(m => m.role !== 'system')
				.map(m => ({
					role: m.role === 'assistant' ? 'model' : 'user',
					parts: [{ text: m.content }],
				}));

			const response = await fetch(
				`https://generativelanguage.googleapis.com/v1beta/models/${this.model}:streamGenerateContent?alt=sse&key=${this.apiKey}`,
				{
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						system_instruction: { parts: [{ text: systemInstruction }] },
						contents,
						generationConfig: { temperature: 0.3, maxOutputTokens: 2048 },
					}),
				}
			);

			if (!response.ok) {
				const errorBody = await response.text();
				throw new Error(`Gemini API error ${response.status}: ${errorBody.substring(0, 200)}`);
			}

			const reader = response.body?.getReader();
			if (!reader) throw new Error('No response body');

			const decoder = new TextDecoder();
			let fullText = '';
			let buffer = '';

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n');
				buffer = lines.pop() || '';

				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;
					const data = line.slice(6).trim();
					if (!data) continue;

					try {
						const parsed = JSON.parse(data);
						const token = parsed.candidates?.[0]?.content?.parts?.[0]?.text;
						if (token) {
							fullText += token;
							callbacks.onToken(token);
						}
					} catch { /* skip */ }
				}
			}

			callbacks.onDone(fullText);
		} catch (error) {
			callbacks.onError(error instanceof Error ? error : new Error(String(error)));
		}
	}

	async chatSync(messages: LLMMessage[]): Promise<string> {
		return new Promise((resolve, reject) => {
			let result = '';
			this.chat(messages, {
				onToken: () => {},
				onDone: (text) => resolve(text),
				onError: (err) => reject(err),
			});
		});
	}

	dispose() {}
}

// ── Ollama Provider (fully local, no API key needed) ──

export class OllamaProvider implements LLMProvider {
	readonly name = 'Ollama';
	private model: string;
	private baseUrl: string;

	constructor(model = 'llama3.2', baseUrl = 'http://localhost:11434') {
		this.model = model;
		this.baseUrl = baseUrl;
	}

	async chat(messages: LLMMessage[], callbacks: LLMStreamCallbacks): Promise<void> {
		try {
			const response = await fetch(`${this.baseUrl}/api/chat`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: this.model,
					messages,
					stream: true,
					options: { temperature: 0.3 },
				}),
			});

			if (!response.ok) throw new Error(`Ollama error ${response.status}`);

			const reader = response.body?.getReader();
			if (!reader) throw new Error('No response body');

			const decoder = new TextDecoder();
			let fullText = '';
			let buffer = '';

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n');
				buffer = lines.pop() || '';

				for (const line of lines) {
					if (!line.trim()) continue;
					try {
						const parsed = JSON.parse(line);
						const token = parsed.message?.content;
						if (token) {
							fullText += token;
							callbacks.onToken(token);
						}
					} catch { /* skip */ }
				}
			}

			callbacks.onDone(fullText);
		} catch (error) {
			callbacks.onError(error instanceof Error ? error : new Error(String(error)));
		}
	}

	async chatSync(messages: LLMMessage[]): Promise<string> {
		const response = await fetch(`${this.baseUrl}/api/chat`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				model: this.model,
				messages,
				stream: false,
				options: { temperature: 0.3 },
			}),
		});

		if (!response.ok) throw new Error(`Ollama error ${response.status}`);
		const data = await response.json();
		return data.message.content;
	}

	dispose() {}
}

// ── Factory ──

export type LLMProviderType = 'openai' | 'gemini' | 'ollama';

export function createLLMProvider(
	type: LLMProviderType,
	config: { apiKey?: string; model?: string; baseUrl?: string }
): LLMProvider {
	switch (type) {
		case 'openai':
			return new OpenAIProvider(config.apiKey || '', config.model || 'gpt-4o-mini', config.baseUrl);
		case 'gemini':
			return new GeminiProvider(config.apiKey || '', config.model || 'gemini-2.0-flash');
		case 'ollama':
			return new OllamaProvider(config.model || 'llama3.2', config.baseUrl || 'http://localhost:11434');
		default:
			throw new Error(`Unknown LLM provider: ${type}`);
	}
}
