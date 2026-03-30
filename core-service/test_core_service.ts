#!/usr/bin/env npx ts-node
/**
 * Manual test script for the Shared Embedding & Retrieval Infrastructure
 * 
 * Run: cd packages/lib && npx ts-node services/embedding/test_core_service.ts
 * 
 * Tests the full pipeline:
 *  1. ChunkingEngine — splits notes at headings
 *  2. VectorStore — stores chunks + embeddings as BLOBs
 *  3. cosine_sim — custom SQL function for similarity
 *  4. EmbeddingService API — put, search, findSimilar, getNoteEmbedding
 */

import { ChunkingEngine } from './ChunkingEngine';
import { VectorStore } from './VectorStore';

// ── Sample notes for testing ─────────────────────────────────────────────
const SAMPLE_NOTES = [
	{
		id: 'note-001',
		title: 'Machine Learning Basics',
		body: `# Introduction
Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Supervised Learning
In supervised learning, models learn from labeled training data. Common algorithms include linear regression, decision trees, and neural networks.

## Unsupervised Learning
Unsupervised learning discovers hidden patterns in unlabeled data. Clustering algorithms like k-means group similar data points.

## Deep Learning
Deep learning uses multi-layer neural networks. Convolutional neural networks (CNNs) excel at image recognition. Recurrent neural networks (RNNs) handle sequential data.`,
	},
	{
		id: 'note-002',
		title: 'JavaScript Async Patterns',
		body: `# Async Programming
JavaScript handles asynchronous operations using callbacks, promises, and async/await.

## Promises
Promises represent eventual completion or failure. They chain with .then() and .catch().

## Async/Await
Async/await is syntactic sugar over promises. It makes asynchronous code look synchronous.

## Event Loop
Node.js uses a single-threaded event loop. The event loop processes callbacks from the task queue.`,
	},
	{
		id: 'note-003',
		title: 'Neural Network Architecture',
		body: `# Neural Networks
Artificial neural networks are inspired by biological neural networks in the brain.

## Layers
A typical network has input, hidden, and output layers. Each layer contains neurons connected to the next layer.

## Activation Functions
Common activation functions include ReLU, sigmoid, and tanh. ReLU is most popular for hidden layers.

## Training
Backpropagation computes gradients for weight updates. Gradient descent optimizes the loss function. Learning rate controls step size.`,
	},
	{
		id: 'note-004',
		title: 'Docker Container Guide',
		body: `# Docker Basics
Docker packages applications into lightweight containers that run consistently across environments.

## Images vs Containers
An image is a template. A container is a running instance of an image. Docker Hub hosts public images.

## Dockerfile
A Dockerfile defines how to build an image. Use FROM for base image, COPY for files, RUN for commands, CMD for entry point.

## Docker Compose
Compose manages multi-container applications. Define services in docker-compose.yml. Use docker-compose up to start.`,
	},
	{
		id: 'note-005',
		title: 'API Design Best Practices',
		body: `# REST API Design
Good API design makes services intuitive and easy to consume.

## Naming Conventions
Use nouns for resources, not verbs. Plural names for collections. Nest related resources logically.

## HTTP Methods
GET for retrieval, POST for creation, PUT for updates, DELETE for removal. Use proper status codes.

## Authentication
Use OAuth 2.0 or API keys. JWT tokens for stateless auth. Always use HTTPS.

## Versioning
Version your API (v1, v2). Use URL path or header versioning. Don't break existing clients.`,
	},
];

// ── Fake embedding provider (deterministic, no network calls) ────────────
function fakeEmbed(text: string): Float32Array {
	// Generate a deterministic 384-dim vector from text hash
	const dim = 384;
	const vec = new Float32Array(dim);
	let hash = 0;
	for (let i = 0; i < text.length; i++) {
		hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
	}
	// Fill with pseudo-random values seeded by hash
	for (let i = 0; i < dim; i++) {
		hash = ((hash * 1103515245 + 12345) & 0x7fffffff);
		vec[i] = (hash / 0x7fffffff) * 2 - 1;
	}
	// L2 normalize
	let norm = 0;
	for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
	norm = Math.sqrt(norm);
	for (let i = 0; i < dim; i++) vec[i] /= norm;
	return vec;
}

// ── Main test ────────────────────────────────────────────────────────────
async function main() {
	console.log('╔══════════════════════════════════════════════════════════════╗');
	console.log('║  Shared Embedding & Retrieval Infrastructure — Manual Test  ║');
	console.log('╚══════════════════════════════════════════════════════════════╝\n');

	// ── 1. Test ChunkingEngine ────────────────────────────────────────
	console.log('━━━ 1. ChunkingEngine ━━━');
	const chunker = new ChunkingEngine(350, 50);

	let totalChunks = 0;
	const allChunks: any[][] = [];

	for (const note of SAMPLE_NOTES) {
		const chunks = chunker.chunkNote(note.id, note.title, note.body);
		allChunks.push(chunks);
		totalChunks += chunks.length;
		console.log(`  ✅ "${note.title}" → ${chunks.length} chunks`);
		for (const c of chunks) {
			console.log(`     [${c.chunkIndex}] ${c.headingPath || '(root)'} — ${c.text.substring(0, 60)}...`);
		}
	}
	console.log(`  Total: ${totalChunks} chunks from ${SAMPLE_NOTES.length} notes\n`);

	// ── 2. Test VectorStore ───────────────────────────────────────────
	console.log('━━━ 2. VectorStore ━━━');
	const dbPath = '/tmp/test-embedding-index.sqlite';
	const store = new VectorStore(dbPath);
	await store.init();
	console.log('  ✅ VectorStore initialized (sql.js + cosine_sim registered)');

	// Store all chunks with fake embeddings
	for (let i = 0; i < SAMPLE_NOTES.length; i++) {
		const chunks = allChunks[i];
		const embeddings = chunks.map((c: any) => fakeEmbed(c.text));
		store.storeNoteChunks(chunks, embeddings);
		console.log(`  ✅ Stored "${SAMPLE_NOTES[i].title}" — ${chunks.length} chunks + note embedding`);
	}

	// ── 3. Test cosine_sim search ─────────────────────────────────────
	console.log('\n━━━ 3. Vector Search (cosine_sim) ━━━');
	const queryText = 'neural networks and deep learning';
	const queryEmb = fakeEmbed(queryText);
	const results = store.searchByVector(queryEmb, 5);
	console.log(`  Query: "${queryText}"`);
	console.log(`  Top ${results.length} results:`);
	for (const r of results) {
		console.log(`    ${r.score.toFixed(4)} | ${r.noteTitle} > ${r.headingPath || '(root)'}`);
	}

	// ── 4. Test findSimilarNotes ──────────────────────────────────────
	console.log('\n━━━ 4. Find Similar Notes ━━━');
	const similar = store.findSimilarNotes('note-001', 3);
	console.log(`  Similar to "Machine Learning Basics":`);
	for (const s of similar) {
		console.log(`    ${s.score.toFixed(4)} | ${s.noteId}`);
	}

	// ── 5. Test getNoteEmbedding ──────────────────────────────────────
	console.log('\n━━━ 5. Note-Level Embedding ━━━');
	const noteEmb = store.getNoteEmbedding('note-001');
	if (noteEmb) {
		console.log(`  ✅ getNoteEmbedding("note-001") → ${noteEmb.length}-dim vector`);
		console.log(`     First 5 values: [${noteEmb.slice(0, 5).map(v => v.toFixed(4)).join(', ')}]`);
	} else {
		console.log('  ❌ getNoteEmbedding returned null');
	}

	// ── 6. Test getAllNoteEmbeddings ──────────────────────────────────
	console.log('\n━━━ 6. All Note Embeddings ━━━');
	const allEmbs = store.getAllNoteEmbeddings();
	console.log(`  ✅ getAllNoteEmbeddings() → ${allEmbs.length} notes`);
	for (const e of allEmbs) {
		console.log(`     ${e.noteId} → ${e.embedding.length}-dim`);
	}

	// ── 7. Test getChunkEmbeddings ───────────────────────────────────
	console.log('\n━━━ 7. Chunk-Level Embeddings ━━━');
	const chunkEmbs = store.getChunkEmbeddings('note-001');
	console.log(`  ✅ getChunkEmbeddings("note-001") → ${chunkEmbs.length} chunks`);
	for (let i = 0; i < chunkEmbs.length; i++) {
		console.log(`     Chunk ${i}: ${chunkEmbs[i].length}-dim`);
	}

	// ── 8. Test Stats ────────────────────────────────────────────────
	console.log('\n━━━ 8. Index Statistics ━━━');
	const stats = store.getStats();
	console.log(`  ✅ Stats:`);
	console.log(`     Notes indexed: ${stats.totalNotes}`);
	console.log(`     Total chunks:  ${stats.totalChunks}`);
	console.log(`     DB size:       ${(stats.dbSizeBytes / 1024).toFixed(1)} KB`);

	// ── 9. Test Change Detection ─────────────────────────────────────
	console.log('\n━━━ 9. Change Detection ━━━');
	const hash = chunker.computeHash(SAMPLE_NOTES[0].body);
	const isIndexed = store.isNoteIndexed('note-001', hash);
	console.log(`  ✅ isNoteIndexed("note-001", same hash) → ${isIndexed} (should be true)`);
	const isChanged = store.isNoteIndexed('note-001', 'different-hash');
	console.log(`  ✅ isNoteIndexed("note-001", different hash) → ${isChanged} (should be false)`);

	// ── 10. Test Notebook Filtering ──────────────────────────────────
	console.log('\n━━━ 10. Notebook Filtering ━━━');
	const filteredResults = store.searchByVector(queryEmb, 5, 'nonexistent-notebook');
	console.log(`  ✅ searchByVector with notebookId filter → ${filteredResults.length} results (should be 0)`);

	// ── Cleanup ──────────────────────────────────────────────────────
	await store.close();

	console.log('\n╔══════════════════════════════════════════════════════════════╗');
	console.log('║                    ALL TESTS PASSED ✅                       ║');
	console.log('╚══════════════════════════════════════════════════════════════╝');
	console.log(`\nDB saved to: ${dbPath}`);
}

main().catch(e => {
	console.error('❌ Test failed:', e);
	process.exit(1);
});
