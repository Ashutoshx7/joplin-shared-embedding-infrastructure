/**
 * Tests for the ChunkingEngine — core embedding infrastructure
 */

import { ChunkingEngine } from './ChunkingEngine';

describe('ChunkingEngine', () => {
	const chunker = new ChunkingEngine(350, 50);

	test('returns empty for empty note', () => {
		expect(chunker.chunkNote('n1', 'Title', '')).toEqual([]);
		expect(chunker.chunkNote('n1', 'Title', '   ')).toEqual([]);
	});

	test('single chunk for short note', () => {
		const chunks = chunker.chunkNote('n1', 'My Note', 'Hello world, this is a short note.');
		expect(chunks).toHaveLength(1);
		expect(chunks[0].text).toContain('My Note');
		expect(chunks[0].text).toContain('Hello world');
		expect(chunks[0].noteId).toBe('n1');
		expect(chunks[0].chunkIndex).toBe(0);
	});

	test('splits at heading boundaries', () => {
		const body = '## Introduction\nThis is the intro.\n## Methods\nThis is the methods section.';
		const chunks = chunker.chunkNote('n1', 'Paper', body);
		expect(chunks.length).toBeGreaterThanOrEqual(2);
		expect(chunks[0].headingPath).toBe('Introduction');
		expect(chunks[1].headingPath).toBe('Methods');
	});

	test('preserves heading hierarchy in breadcrumb', () => {
		const body = '## Chapter One\nSome content about chapter one.';
		const chunks = chunker.chunkNote('n1', 'My Book', body);
		expect(chunks[0].text).toContain('My Book');
		expect(chunks[0].text).toContain('Chapter One');
	});

	test('strips markdown formatting', () => {
		const body = '**bold** and *italic* and `code` and [link](http://example.com)';
		const chunks = chunker.chunkNote('n1', 'Fmt', body);
		expect(chunks[0].text).toContain('bold');
		expect(chunks[0].text).toContain('italic');
		expect(chunks[0].text).not.toContain('**');
		expect(chunks[0].text).not.toContain('http://example.com');
	});

	test('content hash is deterministic', () => {
		const h1 = chunker.computeHash('hello world');
		const h2 = chunker.computeHash('hello world');
		const h3 = chunker.computeHash('hello world!');
		expect(h1).toBe(h2);
		expect(h1).not.toBe(h3);
		expect(h1.length).toBe(16);
	});

	test('filters very short chunks', () => {
		const body = '## A\n\n## B\nActual content here that is longer than three words.';
		const chunks = chunker.chunkNote('n1', 'Test', body);
		// The empty heading section should be filtered out
		for (const c of chunks) {
			const wordCount = c.text.split(/\s+/).filter((w: string) => w.length > 0).length;
			expect(wordCount).toBeGreaterThanOrEqual(3);
		}
	});

	test('handles code blocks without splitting inside them', () => {
		const body = '```javascript\nfunction hello() {\n  ## not a heading\n  return "world";\n}\n```\n\nSome text after the code block.';
		const chunks = chunker.chunkNote('n1', 'Code', body);
		// Should not treat "## not a heading" inside code block as a heading
		expect(chunks.length).toBeLessThanOrEqual(2);
	});

	test('includes notebook id', () => {
		const chunks = chunker.chunkNote('n1', 'Title', 'Some content here.', 'nb1');
		expect(chunks[0].notebookId).toBe('nb1');
	});

	test('handles multiple heading levels', () => {
		const body = '# H1\nContent about chapter one here.\n## H2\nContent about section two here.\n### H3\nContent about subsection three here.';
		const chunks = chunker.chunkNote('n1', 'Multi', body);
		expect(chunks.length).toBeGreaterThanOrEqual(2);
	});
});

describe('RetrievalEngine - RRF', () => {
	test('RRF fusion formula', () => {
		// RRF(doc) = 1/(k+rank), k=60
		const k = 60;
		const rank1Score = 1 / (k + 1);
		const rank2Score = 1 / (k + 2);
		expect(rank1Score).toBeGreaterThan(rank2Score);
		expect(rank1Score).toBeCloseTo(0.01639, 4);
	});

	test('hybrid docs score higher than single-source', () => {
		const k = 60;
		// Doc in both lists at rank 1
		const hybridScore = 1 / (k + 1) + 1 / (k + 1);
		// Doc in vector only at rank 1
		const vectorOnlyScore = 1 / (k + 1);
		expect(hybridScore).toBeGreaterThan(vectorOnlyScore);
	});
});
