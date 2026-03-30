/**
 * Chat With Notes — Joplin Plugin
 * 
 * A RAG-powered chat interface that lets users ask questions about their
 * Joplin notes and get grounded answers with citations.
 *
 * This plugin is a CONSUMER of the shared embedding infrastructure.
 * It contains ZERO embedding, chunking, or vector storage code.
 * All retrieval is done via: joplin.commands.execute('aiSearch.search', query)
 *
 * Architecture:
 *   User query → Shared Infra search() → PromptBuilder → LLM streaming → Chat UI
 */

import joplin from 'api';
import { SettingItemType } from 'api/types';
import { ChatService } from './ChatService';
import { ConversationStore } from './ConversationStore';
import { LLMProviderType } from './LLMProvider';

const path = require('path');

let chatService: ChatService;
let panelHandle: string;

joplin.plugins.register({
	onStart: async function() {
		const dataDir = await joplin.plugins.dataDir();

		// ── Register Settings ──

		await joplin.settings.registerSection('aiChat', {
			label: 'AI Chat',
			iconName: 'fas fa-comments',
			description: 'Chat with your notes using AI. Requires the AI Search infrastructure plugin.',
		});

		await joplin.settings.registerSettings({
			'aiChat.provider': {
				value: 'ollama',
				type: SettingItemType.String,
				isEnum: true,
				options: {
					'ollama': 'Ollama (Local — free)',
					'openai': 'OpenAI',
					'gemini': 'Gemini (Google — free tier)',
				},
				section: 'aiChat',
				label: 'LLM Provider',
				description: 'Which AI model to use for chat responses',
				public: true,
			},
			'aiChat.model': {
				value: 'llama3.2',
				type: SettingItemType.String,
				section: 'aiChat',
				label: 'Model Name',
				description: 'Ollama: llama3.2, gemma2. OpenAI: gpt-4o-mini. Gemini: gemini-2.0-flash',
				public: true,
			},
			'aiChat.apiKey': {
				value: '',
				type: SettingItemType.String,
				section: 'aiChat',
				label: 'API Key',
				description: 'Required for OpenAI and Gemini. Not needed for Ollama (local).',
				public: true,
				secure: true,
			},
			'aiChat.ollamaUrl': {
				value: 'http://localhost:11434',
				type: SettingItemType.String,
				section: 'aiChat',
				label: 'Ollama URL',
				description: 'Base URL for local Ollama server',
				public: true,
			},
			'aiChat.maxContextChunks': {
				value: 8,
				type: SettingItemType.Int,
				section: 'aiChat',
				label: 'Max Context Chunks',
				description: 'Number of note chunks to include as context (higher = more context but slower)',
				public: true,
				minimum: 1,
				maximum: 20,
			},
			'aiChat.maxHistoryTurns': {
				value: 5,
				type: SettingItemType.Int,
				section: 'aiChat',
				label: 'Conversation History Length',
				description: 'Number of previous turns to include (higher = more memory but slower)',
				public: true,
				minimum: 1,
				maximum: 20,
			},
		});

		// ── Initialize Services ──

		const conversationStore = new ConversationStore(path.join(dataDir, 'conversations.sqlite'));
		await conversationStore.initialize();

		chatService = new ChatService(conversationStore);

		// Configure LLM provider from settings
		await refreshProvider();

		// Check if shared infrastructure is available
		await chatService.checkInfrastructure();

		// ── Create Chat Panel ──

		panelHandle = await joplin.views.panels.create('chatPanel');
		await joplin.views.panels.setHtml(panelHandle, getChatPanelHtml());
		await joplin.views.panels.addScript(panelHandle, './panel.css');

		// ── Panel Message Handling ──

		await joplin.views.panels.onMessage(panelHandle, async (msg: any) => {
			if (msg.type === 'chat') {
				// User sent a message
				const query = msg.query;

				await chatService.chat(
					query,
					// onToken — stream to panel
					(token) => {
						joplin.views.panels.postMessage(panelHandle, {
							type: 'token',
							token,
						});
					},
					// onCitations — send source cards
					(citations) => {
						joplin.views.panels.postMessage(panelHandle, {
							type: 'citations',
							citations,
						});
					},
					// onDone — signal completion
					(fullResponse) => {
						joplin.views.panels.postMessage(panelHandle, {
							type: 'done',
							content: fullResponse,
						});
					},
					// onError — show error
					(error) => {
						joplin.views.panels.postMessage(panelHandle, {
							type: 'error',
							message: error,
						});
					}
				);
			}

			else if (msg.type === 'newConversation') {
				chatService.startNewConversation();
				joplin.views.panels.postMessage(panelHandle, { type: 'cleared' });
			}

			else if (msg.type === 'loadHistory') {
				const history = chatService.getHistory();
				joplin.views.panels.postMessage(panelHandle, {
					type: 'history',
					turns: history,
				});
			}

			else if (msg.type === 'getConversations') {
				const conversations = chatService.getConversations();
				joplin.views.panels.postMessage(panelHandle, {
					type: 'conversationList',
					conversations,
				});
			}

			else if (msg.type === 'loadConversation') {
				chatService.loadConversation(msg.id);
				const history = chatService.getHistory();
				joplin.views.panels.postMessage(panelHandle, {
					type: 'history',
					turns: history,
				});
			}

			else if (msg.type === 'deleteConversation') {
				chatService.deleteConversation(msg.id);
			}

			else if (msg.type === 'navigateToNote') {
				// Click citation → jump to note
				try {
					await joplin.commands.execute('openNote', msg.noteId);
				} catch (e) {
					console.error('Failed to open note:', e);
				}
			}

			else if (msg.type === 'getStatus') {
				const infraReady = await chatService.checkInfrastructure();
				const stats = await chatService.getIndexStats();
				joplin.views.panels.postMessage(panelHandle, {
					type: 'status',
					infraReady,
					provider: chatService.getProviderName(),
					stats,
				});
			}
		});

		// ── Commands ──

		await joplin.commands.register({
			name: 'aiChat.toggle',
			label: 'Toggle AI Chat Panel',
			iconName: 'fas fa-comments',
			execute: async () => {
				const visible = await joplin.views.panels.visible(panelHandle);
				await joplin.views.panels.show(panelHandle, !visible);
			},
		});

		await joplin.views.toolbarButtons.create(
			'aiChatToggle',
			'aiChat.toggle',
			'ToolbarButtonLocation.NoteToolbar' as any
		);

		// Refresh provider when settings change
		await joplin.settings.onChange(async () => {
			await refreshProvider();
		});
	},
});

async function refreshProvider() {
	const providerType = await joplin.settings.value('aiChat.provider') as LLMProviderType;
	const model = await joplin.settings.value('aiChat.model');
	const apiKey = await joplin.settings.value('aiChat.apiKey');
	const ollamaUrl = await joplin.settings.value('aiChat.ollamaUrl');

	chatService.setProvider(providerType, {
		apiKey,
		model,
		baseUrl: providerType === 'ollama' ? ollamaUrl : undefined,
	});
}

function getChatPanelHtml(): string {
	return `<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<style>
		* { margin: 0; padding: 0; box-sizing: border-box; }
		
		body {
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
			background: var(--joplin-background-color, #1e1e2e);
			color: var(--joplin-color, #cdd6f4);
			height: 100vh;
			display: flex;
			flex-direction: column;
		}

		/* ── Header ── */
		.header {
			display: flex;
			align-items: center;
			justify-content: space-between;
			padding: 10px 14px;
			border-bottom: 1px solid var(--joplin-divider-color, #313244);
			background: var(--joplin-background-color, #181825);
		}
		.header h2 {
			font-size: 14px;
			font-weight: 600;
			color: var(--joplin-color, #cba6f7);
		}
		.header-actions {
			display: flex;
			gap: 8px;
		}
		.header-actions button {
			background: none;
			border: 1px solid var(--joplin-divider-color, #45475a);
			color: var(--joplin-color, #a6adc8);
			border-radius: 6px;
			padding: 4px 10px;
			font-size: 12px;
			cursor: pointer;
			transition: all 0.15s;
		}
		.header-actions button:hover {
			background: var(--joplin-background-color-hover, #313244);
			color: var(--joplin-color, #cdd6f4);
		}

		/* ── Status Bar ── */
		.status-bar {
			padding: 6px 14px;
			font-size: 11px;
			color: var(--joplin-color-faded, #6c7086);
			border-bottom: 1px solid var(--joplin-divider-color, #313244);
			display: flex;
			justify-content: space-between;
		}
		.status-dot {
			display: inline-block;
			width: 6px;
			height: 6px;
			border-radius: 50%;
			margin-right: 4px;
		}
		.status-dot.ready { background: #a6e3a1; }
		.status-dot.offline { background: #f38ba8; }

		/* ── Messages ── */
		.messages {
			flex: 1;
			overflow-y: auto;
			padding: 14px;
			display: flex;
			flex-direction: column;
			gap: 12px;
		}
		.message {
			max-width: 92%;
			padding: 10px 14px;
			border-radius: 12px;
			font-size: 13px;
			line-height: 1.5;
			word-wrap: break-word;
		}
		.message.user {
			align-self: flex-end;
			background: var(--joplin-selected-color, #45475a);
			color: var(--joplin-color, #cdd6f4);
			border-bottom-right-radius: 4px;
		}
		.message.assistant {
			align-self: flex-start;
			background: var(--joplin-background-color3, #1e1e2e);
			border: 1px solid var(--joplin-divider-color, #313244);
			border-bottom-left-radius: 4px;
		}
		.message.error {
			align-self: center;
			background: rgba(243, 139, 168, 0.15);
			border: 1px solid rgba(243, 139, 168, 0.3);
			color: #f38ba8;
			font-size: 12px;
		}

		/* ── Citations ── */
		.citations {
			display: flex;
			flex-wrap: wrap;
			gap: 6px;
			margin-top: 8px;
		}
		.citation-card {
			background: var(--joplin-background-color, #181825);
			border: 1px solid var(--joplin-divider-color, #313244);
			border-radius: 8px;
			padding: 6px 10px;
			font-size: 11px;
			cursor: pointer;
			transition: all 0.15s;
			max-width: 200px;
		}
		.citation-card:hover {
			background: var(--joplin-background-color-hover, #313244);
			border-color: #89b4fa;
		}
		.citation-title {
			font-weight: 600;
			color: #89b4fa;
			white-space: nowrap;
			overflow: hidden;
			text-overflow: ellipsis;
		}
		.citation-snippet {
			color: var(--joplin-color-faded, #6c7086);
			margin-top: 2px;
			display: -webkit-box;
			-webkit-line-clamp: 2;
			-webkit-box-orient: vertical;
			overflow: hidden;
		}
		.citation-score {
			color: #a6e3a1;
			font-size: 10px;
			margin-top: 2px;
		}

		/* ── Streaming indicator ── */
		.typing-indicator {
			display: none;
			align-self: flex-start;
			padding: 8px 14px;
		}
		.typing-indicator.active { display: flex; }
		.typing-indicator span {
			width: 6px;
			height: 6px;
			background: #89b4fa;
			border-radius: 50%;
			margin: 0 2px;
			animation: bounce 1.4s infinite ease-in-out;
		}
		.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
		.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
		@keyframes bounce {
			0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
			40% { transform: scale(1); opacity: 1; }
		}

		/* ── Input ── */
		.input-area {
			padding: 10px 14px;
			border-top: 1px solid var(--joplin-divider-color, #313244);
			display: flex;
			gap: 8px;
		}
		.input-area textarea {
			flex: 1;
			background: var(--joplin-background-color3, #1e1e2e);
			border: 1px solid var(--joplin-divider-color, #45475a);
			color: var(--joplin-color, #cdd6f4);
			border-radius: 8px;
			padding: 8px 12px;
			font-size: 13px;
			font-family: inherit;
			resize: none;
			min-height: 38px;
			max-height: 120px;
			outline: none;
			transition: border-color 0.15s;
		}
		.input-area textarea:focus {
			border-color: #89b4fa;
		}
		.input-area button {
			background: #89b4fa;
			color: #1e1e2e;
			border: none;
			border-radius: 8px;
			padding: 8px 16px;
			font-size: 13px;
			font-weight: 600;
			cursor: pointer;
			transition: all 0.15s;
			white-space: nowrap;
		}
		.input-area button:hover { background: #74c7ec; }
		.input-area button:disabled {
			opacity: 0.5;
			cursor: not-allowed;
		}

		/* ── Welcome ── */
		.welcome {
			flex: 1;
			display: flex;
			flex-direction: column;
			align-items: center;
			justify-content: center;
			text-align: center;
			padding: 40px;
			color: var(--joplin-color-faded, #6c7086);
		}
		.welcome h3 {
			font-size: 16px;
			color: var(--joplin-color, #cdd6f4);
			margin-bottom: 8px;
		}
		.welcome p {
			font-size: 13px;
			max-width: 280px;
			line-height: 1.5;
		}
	</style>
</head>
<body>
	<div class="header">
		<h2>💬 AI Chat</h2>
		<div class="header-actions">
			<button onclick="newConversation()">+ New</button>
		</div>
	</div>

	<div class="status-bar">
		<span id="infraStatus"><span class="status-dot offline"></span> Checking...</span>
		<span id="providerStatus">—</span>
	</div>

	<div class="messages" id="messages">
		<div class="welcome" id="welcome">
			<h3>Chat with your notes</h3>
			<p>Ask questions about your Joplin notes. Answers are grounded in your actual content with source citations.</p>
		</div>
	</div>

	<div class="typing-indicator" id="typing">
		<span></span><span></span><span></span>
	</div>

	<div class="input-area">
		<textarea 
			id="queryInput" 
			placeholder="Ask about your notes..."
			rows="1"
			onkeydown="handleKeydown(event)"
		></textarea>
		<button id="sendBtn" onclick="sendMessage()">Send</button>
	</div>

	<script>
		let isStreaming = false;
		let currentAssistantMsg = null;

		// ── IPC with plugin backend ──
		function postToPlugin(msg) {
			webviewApi.postMessage(msg);
		}

		// ── Send message ──
		function sendMessage() {
			const input = document.getElementById('queryInput');
			const query = input.value.trim();
			if (!query || isStreaming) return;

			// Hide welcome
			const welcome = document.getElementById('welcome');
			if (welcome) welcome.style.display = 'none';

			// Show user message
			addMessage('user', query);
			input.value = '';
			input.style.height = '38px';

			// Show typing indicator
			isStreaming = true;
			document.getElementById('typing').classList.add('active');
			document.getElementById('sendBtn').disabled = true;

			// Create empty assistant message for streaming
			currentAssistantMsg = addMessage('assistant', '');

			// Send to plugin
			postToPlugin({ type: 'chat', query });
		}

		// ── Handle incoming messages from plugin ──
		webviewApi.onMessage(function(msg) {
			if (msg.type === 'token') {
				if (currentAssistantMsg) {
					currentAssistantMsg.textContent += msg.token;
					scrollToBottom();
				}
			}
			else if (msg.type === 'citations') {
				if (currentAssistantMsg && msg.citations.length > 0) {
					const citDiv = document.createElement('div');
					citDiv.className = 'citations';
					msg.citations.forEach(function(c) {
						const card = document.createElement('div');
						card.className = 'citation-card';
						card.onclick = function() { postToPlugin({ type: 'navigateToNote', noteId: c.noteId }); };
						card.innerHTML = 
							'<div class="citation-title">' + escapeHtml(c.noteTitle) + '</div>' +
							'<div class="citation-snippet">' + escapeHtml(c.snippet) + '</div>' +
							'<div class="citation-score">Score: ' + c.score.toFixed(3) + '</div>';
						citDiv.appendChild(card);
					});
					currentAssistantMsg.after(citDiv);
				}
			}
			else if (msg.type === 'done') {
				isStreaming = false;
				document.getElementById('typing').classList.remove('active');
				document.getElementById('sendBtn').disabled = false;
				currentAssistantMsg = null;
				document.getElementById('queryInput').focus();
			}
			else if (msg.type === 'error') {
				isStreaming = false;
				document.getElementById('typing').classList.remove('active');
				document.getElementById('sendBtn').disabled = false;
				addMessage('error', msg.message);
				currentAssistantMsg = null;
			}
			else if (msg.type === 'cleared') {
				document.getElementById('messages').innerHTML = 
					'<div class="welcome" id="welcome">' +
					'<h3>Chat with your notes</h3>' +
					'<p>Ask questions about your Joplin notes.</p></div>';
			}
			else if (msg.type === 'history') {
				const messagesDiv = document.getElementById('messages');
				messagesDiv.innerHTML = '';
				var welcome = document.getElementById('welcome');
				if (welcome) welcome.style.display = 'none';
				msg.turns.forEach(function(t) {
					addMessage(t.role, t.content);
				});
				scrollToBottom();
			}
			else if (msg.type === 'status') {
				var dot = msg.infraReady ? 'ready' : 'offline';
				var label = msg.infraReady ? 'Index ready' : 'Index not available';
				if (msg.stats) label += ' (' + (msg.stats.totalChunks || 0) + ' chunks)';
				document.getElementById('infraStatus').innerHTML = 
					'<span class="status-dot ' + dot + '"></span> ' + label;
				document.getElementById('providerStatus').textContent = msg.provider;
			}
		});

		function addMessage(role, content) {
			var div = document.createElement('div');
			div.className = 'message ' + role;
			div.textContent = content;
			document.getElementById('messages').appendChild(div);
			scrollToBottom();
			return div;
		}

		function scrollToBottom() {
			var m = document.getElementById('messages');
			m.scrollTop = m.scrollHeight;
		}

		function handleKeydown(e) {
			if (e.key === 'Enter' && !e.shiftKey) {
				e.preventDefault();
				sendMessage();
			}
		}

		function newConversation() {
			postToPlugin({ type: 'newConversation' });
		}

		function escapeHtml(str) {
			var div = document.createElement('div');
			div.textContent = str;
			return div.innerHTML;
		}

		// Auto-resize textarea
		document.getElementById('queryInput').addEventListener('input', function() {
			this.style.height = '38px';
			this.style.height = Math.min(this.scrollHeight, 120) + 'px';
		});

		// Request status on load
		postToPlugin({ type: 'getStatus' });
		postToPlugin({ type: 'loadHistory' });
	</script>
</body>
</html>`;
}
