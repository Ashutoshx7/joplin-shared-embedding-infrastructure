# Shared Embedding & Retrieval Infrastructure for Joplin

**GSoC 2026 Proposal — Ashutoshx7**

A unified embedding and retrieval backbone for all Joplin AI projects. One core service that indexes notes, stores vectors, and serves hybrid search — so AI Search, Chat, Categorization, Note Graphs, and Image Labeling don't each rebuild the same pipeline.

## Repository Structure

```
core-service/              ← packages/lib/services/embedding/ (core Joplin service)
├── EmbeddingService.ts    ← Singleton API: put(), search(), embed(), findSimilarNotes()
├── VectorStore.ts         ← sql.js storage, cosine_sim(), BLOB embeddings
├── RetrievalEngine.ts     ← RRF + RSE + decomposition + reranking
├── ChunkingEngine.ts      ← Markdown-aware heading-based chunker
├── EmbeddingProvider.ts   ← Ollama / OpenAI / Local (WASM) providers
└── test_core_service.ts   ← Standalone manual test script

prototype-plugin/          ← Full working Joplin plugin prototype
├── index.ts               ← Plugin entry point + command registration
├── vectorStore.ts         ← sql.js vector store with cosine_sim
├── embeddings.ts          ← Provider abstraction (Local/Ollama/OpenAI)
├── chunker.ts             ← Markdown-aware chunking engine
├── retrieval.ts           ← Hybrid retrieval (RRF + RSE)
├── reranker.ts            ← Cross-encoder reranking
├── queryDecomposer.ts     ← Complex query decomposition
├── indexer.ts             ← Background indexing with progress
├── types.ts               ← Shared type definitions
├── chunker.test.ts        ← Chunker unit tests
└── retrieval.test.ts      ← Retrieval unit tests

sample-consumer/           ← Related Notes sidebar plugin (13 KB, zero deps)
└── index.ts               ← Demonstrates API usage via joplin.commands.execute()
```

## Key Numbers

| Metric | Value |
|---|---|
| Core service | 1,014 lines |
| Prototype plugin | 1,700+ lines |
| Tests | 30 passing |
| Sample consumer | 158 lines, 13 KB .jpl |
| TypeScript errors | 0 |

## Architecture

```
Note updated → ChunkingEngine (heading split, 350 tokens, SHA-256 hash)
→ EmbeddingProvider (Ollama / OpenAI / Local WASM)
→ VectorStore (sql.js + cosine_sim SQL function + BLOB storage)
→ RetrievalEngine (RRF hybrid + RSE + decomposition + reranking)
→ EmbeddingService (singleton API)
→ Consumer plugins call via joplin.commands.execute()
```

## How to Run

### Prototype Plugin
```bash
cd prototype-plugin/../   # In the joplin-ai-search directory
npm install
npm run dist              # Builds .jpl plugin archive
```

### Tests
```bash
npx jest --verbose        # 30 tests pass
```

### Core Service (type check)
```bash
cd core-service/../       # In packages/lib/
npx tsc --noEmit services/embedding/*.ts
```

## API

```typescript
EmbeddingService.instance().put(noteId);              // Index a note
EmbeddingService.instance().search(query, options);    // Hybrid search
EmbeddingService.instance().findSimilarNotes(noteId);  // Related notes
EmbeddingService.instance().embed(text);               // Raw embedding
EmbeddingService.instance().getNoteEmbedding(noteId);  // Note-level vector
EmbeddingService.instance().getAllNoteEmbeddings();     // All note vectors
EmbeddingService.instance().getChunkEmbeddings(noteId);// Chunk-level vectors
EmbeddingService.instance().getStats();                // Index statistics
```

## Links

- [GSoC Proposal Draft](https://discourse.joplinapp.org/t/gsoc-proposal-drafts/)
- [Design Discussion Thread](https://discourse.joplinapp.org/t/design-discussion-shared-embedding-retrieval-infrastructure-for-joplin-ai-features/49356)
- [Joplin](https://github.com/laurent22/joplin)

## License

MIT
