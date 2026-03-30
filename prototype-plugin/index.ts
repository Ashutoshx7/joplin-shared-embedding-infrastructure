/**
 * Plugin entry point
 *
 * Wires together all components: chunker → embedding → vector store →
 * reranker → query decomposer → retrieval engine → search panel UI.
 * Registers event hooks for incremental indexing and commands for
 * the public API.
 *
 * Addresses mentor feedback:
 *  - Model change detection → auto-rebuild (@adamoutler)
 *  - Cost/token estimation before indexing (@adamoutler)
 *  - getNoteEmbedding API command (@Ashutoshx7 proposal)
 *  - Hybrid balance slider (@shikuz, @adamoutler)
 *  - Reranking toggle (@shikuz, @adamoutler)
 *  - Query decomposition toggle (@shikuz)
 */

import joplin from 'api';
import { SettingItemType, ToolbarButtonLocation } from 'api/types';

import { VectorStore } from './vectorStore';
import { createEmbeddingProvider, EmbeddingProvider, ProviderType } from './embeddings';
import { Indexer, IndexProgress } from './indexer';
import { RetrievalEngine, SearchResult } from './retrieval';
import { Reranker } from './reranker';
import { QueryDecomposer } from './queryDecomposer';

const path = require('path');

let vectorStore: VectorStore;
let embeddingProvider: EmbeddingProvider;
let indexer: Indexer;
let retrieval: RetrievalEngine;
let reranker: Reranker;
let decomposer: QueryDecomposer;
let panelHandle: string;

let currentState: 'not-indexed' | 'indexing' | 'ready' = 'not-indexed';
let pollingInterval: ReturnType<typeof setInterval> | null = null;
let debounceTimer: ReturnType<typeof setTimeout> | null = null;

joplin.plugins.register({
	onStart: async function() {
		// ─── Register Settings ─────────────────────────────────────────

		await joplin.settings.registerSection('aiSearch', {
			label: 'AI Search',
			iconName: 'fas fa-brain',
			description: 'Semantic search for your notes using local embeddings.',
		});

		await joplin.settings.registerSettings({
			// ── Core ──────────────────────────
			'aiSearch.enabled': {
				section: 'aiSearch',
				label: 'Enable AI Search',
				type: SettingItemType.Bool,
				value: false,
				public: true,
				description: 'Start building the embedding index when enabled. All AI features are optional.',
			},
			'aiSearch.provider': {
				section: 'aiSearch',
				label: 'Embedding Provider',
				type: SettingItemType.Int,
				value: 1, // Default to Ollama
				isEnum: true,
				options: { 0: 'Local (Transformers.js)', 1: 'Ollama', 2: 'OpenAI' },
				public: true,
				description: 'How to generate embeddings. "Local" runs entirely on your machine.',
			},
			'aiSearch.ollamaEndpoint': {
				section: 'aiSearch',
				label: 'Ollama Endpoint',
				type: SettingItemType.String,
				value: 'http://localhost:11434',
				public: true,
				description: 'Ollama API endpoint (only used if provider is Ollama).',
			},
			'aiSearch.ollamaModel': {
				section: 'aiSearch',
				label: 'Ollama Embedding Model',
				type: SettingItemType.String,
				value: 'nomic-embed-text',
				public: true,
				description: 'Ollama embedding model name.',
			},
			'aiSearch.openaiApiKey': {
				section: 'aiSearch',
				label: 'OpenAI API Key',
				type: SettingItemType.String,
				value: '',
				public: true,
				secure: true,
				description: 'OpenAI API key (only used if provider is OpenAI).',
			},

			// ── Hybrid Balance (@adamoutler, @shikuz) ──────────────
			'aiSearch.hybridBalance': {
				section: 'aiSearch',
				label: 'Search Balance (0=Keyword, 100=Semantic)',
				type: SettingItemType.Int,
				value: 50,
				minimum: 0,
				maximum: 100,
				step: 10,
				public: true,
				description: '0 = keyword search only, 100 = vector search only, 50 = balanced hybrid. Adjust to your preference.',
			},

			// ── Reranking (@shikuz, @adamoutler) ───────────────────
			'aiSearch.rerankEnabled': {
				section: 'aiSearch',
				label: 'Enable Reranking',
				type: SettingItemType.Bool,
				value: false,
				public: true,
				description: 'Re-score top results using a cross-encoder model for higher precision. Uses more compute.',
			},
			'aiSearch.rerankModel': {
				section: 'aiSearch',
				label: 'Reranking Model',
				type: SettingItemType.String,
				value: 'llama3.2:1b',
				public: true,
				description: 'Ollama model for reranking (small model recommended).',
			},

			// ── Query Decomposition (@shikuz) ──────────────────────
			'aiSearch.decomposeEnabled': {
				section: 'aiSearch',
				label: 'Enable Query Decomposition',
				type: SettingItemType.Bool,
				value: true,
				public: true,
				description: 'Automatically split complex queries into sub-queries for better retrieval.',
			},
		});

		// ─── Create Search Panel ───────────────────────────────────────

		panelHandle = await joplin.views.panels.create('ai-search-panel');

		// Handle messages from the panel webview FIRST
		await joplin.views.panels.onMessage(panelHandle, async (message: any) => {
			if (message.type === 'search') {
				return handleSearch(message.query);
			} else if (message.type === 'buildIndex') {
				await handleBuildIndex();
				return { done: true };
			} else if (message.type === 'openNote') {
				await joplin.commands.execute('openNote', message.noteId);
			} else if (message.type === 'getState') {
				return { state: currentState };
			} else if (message.type === 'estimateCost') {
				return estimateIndexingCost();
			}
		});

		// ─── Auto-Initialize if Enabled ────────────────────────────────
		// Do this BEFORE setting HTML so we know what state to show

		const enabled = await joplin.settings.value('aiSearch.enabled');
		if (enabled) {
			await initializeEngine();
		}

		// Now we know the actual state — generate HTML accordingly
		const showReady = currentState === 'ready';

		const panelHtml = `
			<div class="container">
				<div id="loading-view" class="state-view" style="display:none">
					<div class="spinner"></div>
					<p>Loading AI Search…</p>
				</div>

				<div id="not-indexed-view" class="state-view" style="display:${showReady ? 'none' : 'block'}">
					<h3>🧠 AI Search</h3>
					<p>Build a semantic index of your notes to enable natural language search.</p>
					<div id="cost-estimate" class="hint" style="margin-bottom:12px;"></div>
					<button id="build-index-btn" class="btn" onclick="
						if(typeof webviewApi !== 'undefined') {
							webviewApi.postMessage({ type: 'estimateCost' }).then(function(est) {
								if (est && est.noteCount > 0) {
									var msg = est.noteCount + ' notes · ~' + est.estimatedTokens + ' tokens';
									if (est.estimatedCost) msg += ' · ~$' + est.estimatedCost;
									msg += ' · ~' + est.estimatedTime;
									document.getElementById('cost-estimate').textContent = msg;
								}
							});
							webviewApi.postMessage({ type: 'buildIndex' });
							document.getElementById('not-indexed-view').style.display='none';
							document.getElementById('indexing-view').style.display='block';
						}
					">Build Index</button>
				</div>

				<div id="indexing-view" class="state-view" style="display:none">
					<div class="spinner"></div>
					<h3>Building Index…</h3>
					<p id="indexing-status">Connecting to embedding provider…</p>
					<div class="progress-bar"><div id="indexing-progress" class="progress-fill" style="width:0%"></div></div>
					<p id="indexing-detail" class="hint"></p>
				</div>

				<div id="error-view" class="state-view" style="display:none">
					<h3>❌ Error</h3>
					<p id="error-message" style="color:#f38ba8;font-size:12px;word-break:break-all;"></p>
					<button id="retry-btn" class="btn" style="margin-top:12px;" onclick="
						if(typeof webviewApi !== 'undefined') webviewApi.postMessage({ type: 'buildIndex' });
						document.getElementById('error-view').style.display='none';
						document.getElementById('indexing-view').style.display='block';
					">Retry</button>
				</div>

				<div id="ready-view" style="display:${showReady ? 'block' : 'none'}">
					<div class="search-box">
						<span class="search-icon">🔍</span>
						<input type="text" id="search-input" placeholder="Search your notes semantically…" autocomplete="off" />
					</div>
					<div id="results"></div>
				</div>

				<div id="status-bar"></div>
			</div>
		`;
		await joplin.views.panels.setHtml(panelHandle, panelHtml);

		// Now add scripts which will successfully find the DOM elements
		await joplin.views.panels.addScript(panelHandle, './panel/panel.css');
		await joplin.views.panels.addScript(panelHandle, './panel/panel.js');

		// ─── Register Commands (Public API) ────────────────────────────

		await joplin.commands.register({
			name: 'aiSearch.toggle',
			label: 'Toggle AI Search Panel',
			iconName: 'fas fa-brain',
			execute: async () => {
				const isVisible = await joplin.views.panels.visible(panelHandle);
				await joplin.views.panels.show(panelHandle, !isVisible);
			},
		});

		await joplin.commands.register({
			name: 'aiSearch.query',
			label: 'AI Search: Query',
			execute: async (...args: any[]) => {
				if (!retrieval) return [];
				const query = args[0] as string;
				const options = args[1] || {};
				return retrieval.search(query, options);
			},
		});

		await joplin.commands.register({
			name: 'aiSearch.findSimilar',
			label: 'AI Search: Find Similar Notes',
			execute: async (...args: any[]) => {
				if (!retrieval) return [];
				const noteId = args[0] as string;
				const limit = args[1] || 5;
				return retrieval.findSimilarNotes(noteId, limit);
			},
		});

		// ── Gap #5: getNoteEmbedding command ────────────────────────
		await joplin.commands.register({
			name: 'aiSearch.getNoteEmbedding',
			label: 'AI Search: Get Note Embedding',
			execute: async (...args: any[]) => {
				if (!vectorStore) return null;
				const noteId = args[0] as string;
				const embedding = vectorStore.getNoteEmbedding(noteId);
				// Return as regular array for JSON serialization
				return embedding ? Array.from(embedding) : null;
			},
		});

		await joplin.commands.register({
			name: 'aiSearch.getAllNoteEmbeddings',
			label: 'AI Search: Get All Note Embeddings',
			execute: async () => {
				if (!vectorStore) return [];
				return vectorStore.getAllNoteEmbeddings().map(ne => ({
					noteId: ne.noteId,
					embedding: Array.from(ne.embedding),
				}));
			},
		});

		await joplin.commands.register({
			name: 'aiSearch.stats',
			label: 'AI Search: Get Stats',
			execute: async () => {
				if (!vectorStore) return null;
				return vectorStore.getStats();
			},
		});

		// ── put(noteId) — @shikuz's core interface ──────────────────
		await joplin.commands.register({
			name: 'aiSearch.put',
			label: 'AI Search: Index/Re-index a Note',
			execute: async (...args: any[]) => {
				if (!indexer) return;
				const noteId = args[0] as string;
				await indexer.reindexNote(noteId);
			},
		});

		// ── embed(text) — arbitrary text embedding for consumer plugins ─
		await joplin.commands.register({
			name: 'aiSearch.embed',
			label: 'AI Search: Embed Text',
			execute: async (...args: any[]) => {
				if (!embeddingProvider) return null;
				const text = args[0] as string;
				const embedding = await embeddingProvider.embed(text);
				return Array.from(embedding);
			},
		});

		// ── getChunkEmbeddings(noteId) — chunk-level vectors ────────
		await joplin.commands.register({
			name: 'aiSearch.getChunkEmbeddings',
			label: 'AI Search: Get Chunk Embeddings',
			execute: async (...args: any[]) => {
				if (!vectorStore) return [];
				const noteId = args[0] as string;
				return vectorStore.getChunkEmbeddings(noteId).map(e => Array.from(e));
			},
		});

		// ── isReady() — check if index is ready for queries ─────────
		await joplin.commands.register({
			name: 'aiSearch.isReady',
			label: 'AI Search: Is Ready',
			execute: async () => {
				return currentState === 'ready';
			},
		});

		await joplin.commands.register({
			name: 'aiSearch.reindex',
			label: 'AI Search: Rebuild Index',
			execute: async () => {
				return handleBuildIndex();
			},
		});

		// ─── Toolbar Button ────────────────────────────────────────────

		await joplin.views.toolbarButtons.create(
			'aiSearchButton',
			'aiSearch.toggle',
			ToolbarButtonLocation.NoteToolbar,
		);

		// ─── Watch for Settings Changes ────────────────────────────────

		await joplin.settings.onChange(async (event: any) => {
			if (event.keys.includes('aiSearch.enabled')) {
				const enabled = await joplin.settings.value('aiSearch.enabled');
				if (enabled) {
					await initializeEngine();
				} else {
					await shutdownEngine();
				}
			}
		});
	},
});

// ─── Core Logic ───────────────────────────────────────────────────────────

async function initializeEngine(): Promise<void> {
	const dataDir = await joplin.plugins.dataDir();
	const installDir = await joplin.plugins.installationDir();

	// Create vector store (sql.js backed)
	const dbPath = path.join(dataDir, 'embedding_index.db');
	vectorStore = new VectorStore(dbPath);
	await vectorStore.init(installDir);

	// Create embedding provider based on settings
	const providerType = await joplin.settings.value('aiSearch.provider');
	const providerMap: Record<number, ProviderType> = { 0: 'local', 1: 'ollama', 2: 'openai' };

	embeddingProvider = createEmbeddingProvider(providerMap[providerType] || 'local', {
		cacheDir: path.join(dataDir, 'models'),
		endpoint: await joplin.settings.value('aiSearch.ollamaEndpoint'),
		modelId: providerType === 1
			? await joplin.settings.value('aiSearch.ollamaModel')
			: undefined,
		apiKey: providerType === 2
			? await joplin.settings.value('aiSearch.openaiApiKey')
			: undefined,
	});

	// ── Gap #4: Model change detection (@adamoutler) ──────────────
	const storedModel = vectorStore.getMeta('model_name');
	const storedDimensions = vectorStore.getMeta('dimensions');
	const currentModel = embeddingProvider.name;
	const currentDimensions = String(embeddingProvider.dimensions);

	if (storedModel && storedModel !== currentModel) {
		console.warn(`AI Search: Model changed from "${storedModel}" to "${currentModel}" — clearing index for rebuild`);
		vectorStore.clear();
		joplin.views.panels.postMessage(panelHandle, {
			type: 'error',
			data: { message: `Embedding model changed (${storedModel} → ${currentModel}). Index cleared — please rebuild.` },
		});
	} else if (storedDimensions && storedDimensions !== currentDimensions) {
		console.warn(`AI Search: Dimensions changed from ${storedDimensions} to ${currentDimensions} — clearing index`);
		vectorStore.clear();
	}

	// Store current model info
	vectorStore.setMeta('model_name', currentModel);
	vectorStore.setMeta('dimensions', currentDimensions);

	// Create reranker
	const rerankEnabled = await joplin.settings.value('aiSearch.rerankEnabled');
	const rerankModel = await joplin.settings.value('aiSearch.rerankModel');
	reranker = new Reranker({
		enabled: rerankEnabled,
		provider: 'ollama',
		endpoint: await joplin.settings.value('aiSearch.ollamaEndpoint'),
		model: rerankModel,
	});

	// Create query decomposer
	const decomposeEnabled = await joplin.settings.value('aiSearch.decomposeEnabled');
	decomposer = new QueryDecomposer({
		enabled: decomposeEnabled,
		llmAssisted: false, // Heuristic only by default — no extra LLM calls
	});

	// Create indexer with progress callback
	const joplinDataGet = async (p: string[], q?: any) => joplin.data.get(p, q);

	indexer = new Indexer(
		vectorStore,
		embeddingProvider,
		joplinDataGet,
		(progress: IndexProgress) => {
			joplin.views.panels.postMessage(panelHandle, {
				type: 'progress',
				data: progress,
			});
		},
	);

	// Create retrieval engine with reranker and decomposer
	retrieval = new RetrievalEngine(
		vectorStore,
		embeddingProvider,
		joplinDataGet,
		reranker,
		decomposer,
	);

	// Check if index already exists
	const stats = vectorStore.getStats();
	if (stats.totalNotes > 0) {
		currentState = 'ready';
		joplin.views.panels.postMessage(panelHandle, { type: 'state', data: { state: 'ready' } });
		joplin.views.panels.postMessage(panelHandle, { type: 'stats', data: stats });

		// Start incremental polling
		startEventPolling();
		setupNoteChangeHook();
	} else {
		currentState = 'not-indexed';
		joplin.views.panels.postMessage(panelHandle, { type: 'state', data: { state: 'not-indexed' } });
	}
}

async function shutdownEngine(): Promise<void> {
	if (pollingInterval) {
		clearInterval(pollingInterval);
		pollingInterval = null;
	}
	if (vectorStore) {
		await vectorStore.close();
	}
	if (embeddingProvider) {
		embeddingProvider.dispose();
	}
	currentState = 'not-indexed';
}

async function handleSearch(query: string): Promise<SearchResult[]> {
	if (!retrieval || currentState !== 'ready') return [];

	try {
		// Read hybrid balance from settings
		const hybridBalance = (await joplin.settings.value('aiSearch.hybridBalance')) / 100;

		const results = await retrieval.search(query, {
			limit: 15,
			hybridBalance,
		});
		joplin.views.panels.postMessage(panelHandle, {
			type: 'results',
			data: results,
		});
		return results;
	} catch (e) {
		console.error('AI Search failed:', e);
		return [];
	}
}

async function handleBuildIndex(): Promise<void> {
	try {
		// Auto-initialize engine if not yet set up
		if (!indexer) {
			console.info('AI Search: initializing engine for first build...');
			await initializeEngine();
		}
		if (!indexer) {
			console.error('AI Search: engine failed to initialize');
			joplin.views.panels.postMessage(panelHandle, {
				type: 'error',
				data: { message: 'Engine failed to initialize. Check that Ollama is running (ollama serve) and the model is pulled (ollama pull nomic-embed-text).' },
			});
			return;
		}
		if (indexer.running) {
			console.info('AI Search: indexer already running');
			return;
		}

		currentState = 'indexing';
		joplin.views.panels.postMessage(panelHandle, { type: 'state', data: { state: 'indexing' } });

		console.info('AI Search: starting index build...');
		await indexer.indexAll();
		currentState = 'ready';

		// Flush to disk
		await vectorStore.flushToDisk();

		// Send final stats
		const stats = vectorStore.getStats();
		console.info('AI Search: index built!', stats);
		joplin.views.panels.postMessage(panelHandle, { type: 'stats', data: stats });
		joplin.views.panels.postMessage(panelHandle, { type: 'state', data: { state: 'ready' } });

		// Start incremental updates
		startEventPolling();
		setupNoteChangeHook();
	} catch (e) {
		console.error('AI Search: index build failed:', e);
		currentState = 'not-indexed';
		const errMsg = e instanceof Error ? e.message : String(e);
		joplin.views.panels.postMessage(panelHandle, {
			type: 'error',
			data: { message: `Build failed: ${errMsg}` },
		});
	}
}

/**
 * Gap #6: Cost/token estimation before indexing (@adamoutler).
 * Returns estimated note count, tokens, cost (for API providers), and time.
 */
async function estimateIndexingCost(): Promise<{
	noteCount: number;
	estimatedTokens: string;
	estimatedCost: string | null;
	estimatedTime: string;
}> {
	try {
		const joplinDataGet = async (p: string[], q?: any) => joplin.data.get(p, q);

		// Get total note count
		const response = await joplinDataGet(['notes'], { fields: 'id', limit: 1 });
		// Joplin paginated API doesn't give total count directly — estimate
		let noteCount = 0;
		let page = 1;
		let hasMore = true;
		while (hasMore) {
			const r = await joplinDataGet(['notes'], { fields: 'id', limit: 100, page });
			if (!r || !r.items || r.items.length === 0) break;
			noteCount += r.items.length;
			hasMore = r.has_more;
			page++;
			// Cap at 5 pages to keep estimation fast
			if (page > 5) {
				noteCount = Math.round(noteCount * 1.5); // Rough extrapolation
				break;
			}
		}

		// Average ~200 words/note, ~1.3 tokens/word, ~3 chunks/note
		const avgTokensPerNote = 200 * 1.3 * 3; // ~780 tokens
		const totalTokens = noteCount * avgTokensPerNote;

		// Cost estimation (only for API providers)
		const providerType = await joplin.settings.value('aiSearch.provider');
		let estimatedCost: string | null = null;
		if (providerType === 2) {
			// OpenAI text-embedding-3-small: $0.02 per 1M tokens
			const cost = (totalTokens / 1_000_000) * 0.02;
			estimatedCost = cost.toFixed(4);
		}

		// Time estimation
		let estimatedTime = '';
		if (providerType === 0) {
			// Local WASM: ~50ms per chunk, 3 chunks/note
			estimatedTime = `${Math.ceil(noteCount * 3 * 50 / 60000)} min (local)`;
		} else if (providerType === 1) {
			// Ollama: ~20ms per chunk
			estimatedTime = `${Math.ceil(noteCount * 3 * 20 / 60000)} min (Ollama)`;
		} else {
			// OpenAI: ~10ms per chunk (batched)
			estimatedTime = `${Math.ceil(noteCount * 3 * 10 / 60000)} min (API)`;
		}

		return {
			noteCount,
			estimatedTokens: totalTokens > 1_000_000
				? `${(totalTokens / 1_000_000).toFixed(1)}M`
				: `${Math.round(totalTokens / 1000)}K`,
			estimatedCost,
			estimatedTime,
		};
	} catch (e) {
		return { noteCount: 0, estimatedTokens: '0', estimatedCost: null, estimatedTime: 'unknown' };
	}
}

function startEventPolling(): void {
	if (pollingInterval) return;

	// Poll events API every 60 seconds for incremental updates
	pollingInterval = setInterval(async () => {
		if (!indexer) return;
		const count = await indexer.processEvents();
		if (count > 0) {
			const stats = vectorStore.getStats();
			joplin.views.panels.postMessage(panelHandle, { type: 'stats', data: stats });
		}
	}, 60_000);
}

function setupNoteChangeHook(): void {
	// Tier 1: Active note changes (5s debounce)
	joplin.workspace.onNoteChange(async (event: any) => {
		if (!indexer || currentState !== 'ready') return;

		if (debounceTimer) clearTimeout(debounceTimer);
		debounceTimer = setTimeout(async () => {
			await indexer.reindexNote(event.id);
		}, 5000);
	});

	// Tier 2: After sync complete, process any queued events
	joplin.workspace.onSyncComplete(async () => {
		if (!indexer || currentState !== 'ready') return;
		await indexer.processEvents();
	});
}
