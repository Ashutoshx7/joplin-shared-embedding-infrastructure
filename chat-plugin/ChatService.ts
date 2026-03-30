/**
 * ChatService — RAG orchestration engine
 * Retrieves relevant notes → builds prompt → streams LLM response
 * 
 * This is the core: it connects the shared embedding infrastructure
 * (via Joplin commands) to the LLM provider and conversation store.
 * The chat plugin has ZERO embedding code — it's purely a consumer.
 */

import joplin from 'api';
import { LLMProvider, LLMMessage, createLLMProvider, LLMProviderType } from './LLMProvider';
import { PromptBuilder, RetrievedChunk, Citation } from './PromptBuilder';
import { ConversationStore } from './ConversationStore';

export interface ChatResponse {
	conversationId: string;
	content: string;
	citations: Citation[];
	isComplete: boolean;
}

export class ChatService {
	private llmProvider: LLMProvider | null = null;
	private promptBuilder: PromptBuilder;
	private conversationStore: ConversationStore;
	private currentConversationId: string | null = null;
	private isInfraReady = false;

	constructor(conversationStore: ConversationStore) {
		this.promptBuilder = new PromptBuilder();
		this.conversationStore = conversationStore;
	}

	// ── Provider Management ──

	public setProvider(type: LLMProviderType, config: { apiKey?: string; model?: string; baseUrl?: string }): void {
		if (this.llmProvider) this.llmProvider.dispose();
		this.llmProvider = createLLMProvider(type, config);
	}

	public getProviderName(): string {
		return this.llmProvider?.name || 'Not configured';
	}

	// ── Infrastructure Check ──

	public async checkInfrastructure(): Promise<boolean> {
		try {
			this.isInfraReady = await joplin.commands.execute('aiSearch.isReady');
			return this.isInfraReady;
		} catch {
			this.isInfraReady = false;
			return false;
		}
	}

	// ── Conversation Management ──

	public startNewConversation(title = 'New Chat'): string {
		this.currentConversationId = this.conversationStore.createConversation(title);
		return this.currentConversationId;
	}

	public loadConversation(conversationId: string): void {
		this.currentConversationId = conversationId;
	}

	public getCurrentConversationId(): string | null {
		return this.currentConversationId;
	}

	public getConversations() {
		return this.conversationStore.getConversations();
	}

	public getHistory() {
		if (!this.currentConversationId) return [];
		return this.conversationStore.getTurns(this.currentConversationId);
	}

	public deleteConversation(id: string) {
		this.conversationStore.deleteConversation(id);
		if (this.currentConversationId === id) {
			this.currentConversationId = null;
		}
	}

	// ── Core RAG Chat ──

	/**
	 * Send a message and get a streaming response
	 * Pipeline: query → retrieve from shared infra → build prompt → stream LLM
	 */
	public async chat(
		query: string,
		onToken: (token: string) => void,
		onCitations: (citations: Citation[]) => void,
		onDone: (fullResponse: string) => void,
		onError: (error: string) => void
	): Promise<void> {
		if (!this.llmProvider) {
			onError('No LLM provider configured. Go to Tools → Options → AI Chat to set up a provider.');
			return;
		}

		// Auto-create conversation if none exists
		if (!this.currentConversationId) {
			this.startNewConversation();
		}

		// Save user message
		this.conversationStore.addTurn(this.currentConversationId!, 'user', query);

		try {
			// ── Step 1: Retrieve relevant chunks from shared infrastructure ──
			let retrievedChunks: RetrievedChunk[] = [];
			
			if (this.isInfraReady) {
				try {
					const results = await joplin.commands.execute('aiSearch.search', query, { limit: 8, hybrid: true });
					retrievedChunks = (results || []).map((r: any) => ({
						noteId: r.noteId || r.note_id,
						noteTitle: r.noteTitle || r.note_title || 'Untitled',
						headingPath: r.headingPath || r.heading_path || '',
						text: r.text || r.snippet || '',
						score: r.score || 0,
					}));
				} catch (e) {
					// Infra not available — chat without context
					console.warn('Shared infra search failed, chatting without note context:', e);
				}
			}

			// ── Step 2: Extract and send citations ──
			const citations = this.promptBuilder.extractCitations(retrievedChunks);
			onCitations(citations);

			// ── Step 3: Build prompt with context + history ──
			const history = this.conversationStore.getTurnsAsMessages(this.currentConversationId!);
			// Remove the last turn (current query) since we add it in buildMessages
			const previousHistory = history.slice(0, -1);
			const messages = this.promptBuilder.buildMessages(query, retrievedChunks, previousHistory);

			// ── Step 4: Stream LLM response ──
			await this.llmProvider.chat(messages, {
				onToken: (token) => {
					onToken(token);
				},
				onDone: (fullText) => {
					// Save assistant response
					this.conversationStore.addTurn(
						this.currentConversationId!, 
						'assistant', 
						fullText, 
						citations
					);

					// Auto-title conversation from first query
					const convs = this.conversationStore.getConversations();
					const current = convs.find(c => c.id === this.currentConversationId);
					if (current && current.title === 'New Chat') {
						const title = query.length > 50 ? query.substring(0, 47) + '...' : query;
						this.conversationStore.updateConversationTitle(this.currentConversationId!, title);
					}

					onDone(fullText);
				},
				onError: (error) => {
					onError(`LLM error: ${error.message}`);
				},
			});

		} catch (error) {
			const msg = error instanceof Error ? error.message : String(error);
			onError(`Chat error: ${msg}`);
		}
	}

	/**
	 * Get index stats from the shared infrastructure
	 */
	public async getIndexStats(): Promise<any> {
		try {
			return await joplin.commands.execute('aiSearch.getStats');
		} catch {
			return null;
		}
	}

	// ── Lifecycle ──

	public dispose(): void {
		if (this.llmProvider) this.llmProvider.dispose();
		this.conversationStore.dispose();
	}
}
