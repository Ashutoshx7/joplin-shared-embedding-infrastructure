/**
 * Unit tests for the Markdown-aware note chunker
 */

import { chunkNote, NoteChunk } from './chunker';

describe('chunkNote', () => {

	it('returns empty array for empty body and empty title', () => {
		const chunks = chunkNote('note1', '', '', 'nb1');
		expect(chunks).toEqual([]);
	});

	it('creates a single chunk from title when body is empty', () => {
		const chunks = chunkNote('note1', 'My Title', '', 'nb1');
		expect(chunks).toHaveLength(1);
		expect(chunks[0].text).toBe('My Title');
		expect(chunks[0].noteId).toBe('note1');
		expect(chunks[0].chunkIndex).toBe(0);
	});

	it('creates a single chunk for a short note', () => {
		const chunks = chunkNote('note1', 'Title', 'This is a short note body.', 'nb1');
		expect(chunks).toHaveLength(1);
		expect(chunks[0].text).toContain('Title');
		expect(chunks[0].text).toContain('short note body');
	});

	it('splits on headings', () => {
		const body = `
Some intro text that is reasonably long enough to be meaningful.

## Section A

Content for section A goes here, with enough details.

## Section B

Content for section B is also here, with key information.
		`.trim();

		const chunks = chunkNote('note1', 'My Note', body, 'nb1');
		expect(chunks.length).toBeGreaterThanOrEqual(2);

		// Each chunk should contain sections, not mixed content
		const sectionAChunk = chunks.find(c => c.text.includes('Section A'));
		expect(sectionAChunk).toBeDefined();

		const sectionBChunk = chunks.find(c => c.text.includes('Section B'));
		expect(sectionBChunk).toBeDefined();
	});

	it('preserves heading path', () => {
		const body = `
## Getting Started

### Installation

Run npm install.

### Configuration

Edit config.json.
		`.trim();

		const chunks = chunkNote('note1', 'Setup Guide', body, 'nb1');
		const installChunk = chunks.find(c => c.text.includes('npm install'));
		expect(installChunk).toBeDefined();
		expect(installChunk!.headingPath).toContain('Getting Started');
	});

	it('prepends title and heading path to chunk text', () => {
		const body = `
## Features

This is the features section.
		`.trim();

		const chunks = chunkNote('note1', 'README', body, 'nb1');
		expect(chunks[0].text).toMatch(/^README/);
	});

	it('handles notes with only headings and no body content', () => {
		const body = `
# Only Headings

##

### 
		`.trim();

		const chunks = chunkNote('note1', 'Empty Headings', body, 'nb1');
		// Should still create something for the note with "Only Headings" section
		// Or return empty if all sections have insufficient text
		expect(Array.isArray(chunks)).toBe(true);
	});

	it('strips markdown syntax while preserving semantic content', () => {
		const body = `
This has **bold**, *italic*, ~~strikethrough~~, and [links](http://example.com).

Also has ![images](http://img.png) and \`inline code\`.
		`.trim();

		const chunks = chunkNote('note1', 'Formatting', body, 'nb1');
		expect(chunks.length).toBeGreaterThanOrEqual(1);
		const text = chunks[0].rawText;
		expect(text).not.toContain('**');
		expect(text).not.toContain('~~');
		expect(text).toContain('bold');
		expect(text).toContain('italic');
		expect(text).toContain('links');
	});

	it('sets content hash for change detection', () => {
		const chunks = chunkNote('note1', 'Title', 'Some content here.', 'nb1');
		expect(chunks[0].contentHash).toBeDefined();
		expect(chunks[0].contentHash.length).toBeGreaterThan(0);
	});

	it('produces deterministic hashes for same content', () => {
		const chunks1 = chunkNote('note1', 'T', 'Same content.', 'nb1');
		const chunks2 = chunkNote('note2', 'T', 'Same content.', 'nb2');
		expect(chunks1[0].contentHash).toBe(chunks2[0].contentHash);
	});

	it('handles a note with nested headings', () => {
		const body = `
# Main Topic

This is the intro.

## Subtopic A

Details about A.

### Sub-subtopic A1

Even more specific details about A1.

## Subtopic B

Details about B.
		`.trim();

		const chunks = chunkNote('note1', 'Nested', body, 'nb1');
		expect(chunks.length).toBeGreaterThanOrEqual(3);

		// Check heading path construction
		const a1Chunk = chunks.find(c => c.text.includes('A1'));
		if (a1Chunk) {
			expect(a1Chunk.headingPath).toContain('Subtopic A');
		}
	});

	it('assigns sequential chunk indices', () => {
		const body = `
## Section 1

Content 1.

## Section 2

Content 2.

## Section 3

Content 3.
		`.trim();

		const chunks = chunkNote('note1', 'Ordered', body, 'nb1');
		for (let i = 0; i < chunks.length; i++) {
			expect(chunks[i].chunkIndex).toBe(i);
		}
	});

	it('includes noteId and notebookId in every chunk', () => {
		const chunks = chunkNote('n123', 'Title', 'Content here.', 'nb456');
		for (const chunk of chunks) {
			expect(chunk.noteId).toBe('n123');
			expect(chunk.notebookId).toBe('nb456');
			expect(chunk.noteTitle).toBe('Title');
		}
	});

});
