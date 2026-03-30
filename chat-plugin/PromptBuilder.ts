/**
 * Prompt builder for RAG chat
 * Assembles: system instruction + retrieved context + conversation history
 * Manages token budget to prevent context overflow
 */

import { LLMMessage } from './LLMProvider';

export interface RetrievedChunk {
	noteId: string;
	noteTitle: string;
	headingPath: string;
	text: string;
	score: number;
}

const SYSTEM_PROMPT = `You are a helpful AI assistant for Joplin, a note-taking application. You answer questions based ONLY on the user's notes provided below as context.

Rules:
1. Answer exclusively from the provided note excerpts. If the information isn't in the notes, say so clearly.
2. When citing information, mention the note title in **bold** so the user knows the source.
3. If multiple notes are relevant, synthesize information from all of them.
4. Keep answers concise but thorough.
5. If the user asks to summarize, create, or edit notes — explain that you can only search and answer questions about existing notes.
6. Use markdown formatting in your responses for readability.`;

export class PromptBuilder {
	private maxContextTokens: number;
	private maxHistoryTurns: number;

	constructor(maxContextTokens = 3000, maxHistoryTurns = 5) {
		this.maxContextTokens = maxContextTokens;
		this.maxHistoryTurns = maxHistoryTurns;
	}

	/**
	 * Build the full message array for the LLM
	 */
	public buildMessages(
		query: string,
		retrievedChunks: RetrievedChunk[],
		conversationHistory: LLMMessage[]
	): LLMMessage[] {
		const messages: LLMMessage[] = [];

		// 1. System prompt with retrieved context
		const contextBlock = this.buildContextBlock(retrievedChunks);
		messages.push({
			role: 'system',
			content: `${SYSTEM_PROMPT}\n\n--- RELEVANT NOTES ---\n${contextBlock}\n--- END OF NOTES ---`,
		});

		// 2. Trim conversation history to fit token budget
		const trimmedHistory = this.trimHistory(conversationHistory);
		messages.push(...trimmedHistory);

		// 3. Current user query
		messages.push({ role: 'user', content: query });

		return messages;
	}

	/**
	 * Format retrieved chunks as context with source attribution
	 */
	private buildContextBlock(chunks: RetrievedChunk[]): string {
		if (chunks.length === 0) {
			return 'No relevant notes found for this query.';
		}

		let totalWords = 0;
		const contextParts: string[] = [];

		for (const chunk of chunks) {
			const chunkWords = chunk.text.split(/\s+/).length;
			if (totalWords + chunkWords > this.maxContextTokens) break;

			const heading = chunk.headingPath
				? ` > ${chunk.headingPath}`
				: '';
			contextParts.push(
				`<note title="${chunk.noteTitle}${heading}" id="${chunk.noteId}" score="${chunk.score.toFixed(3)}">\n${chunk.text}\n</note>`
			);
			totalWords += chunkWords;
		}

		return contextParts.join('\n\n');
	}

	/**
	 * Keep only the last N turns, trimming oldest first
	 */
	private trimHistory(history: LLMMessage[]): LLMMessage[] {
		if (history.length === 0) return [];

		// Keep last maxHistoryTurns * 2 messages (each turn = user + assistant)
		const maxMessages = this.maxHistoryTurns * 2;
		if (history.length <= maxMessages) return [...history];

		return history.slice(history.length - maxMessages);
	}

	/**
	 * Extract source citations from retrieved chunks for UI display
	 */
	public extractCitations(chunks: RetrievedChunk[]): Citation[] {
		const seen = new Set<string>();
		const citations: Citation[] = [];

		for (const chunk of chunks) {
			if (seen.has(chunk.noteId)) continue;
			seen.add(chunk.noteId);
			citations.push({
				noteId: chunk.noteId,
				noteTitle: chunk.noteTitle,
				headingPath: chunk.headingPath,
				snippet: chunk.text.substring(0, 120) + '...',
				score: chunk.score,
			});
		}

		return citations;
	}
}

export interface Citation {
	noteId: string;
	noteTitle: string;
	headingPath: string;
	snippet: string;
	score: number;
}
