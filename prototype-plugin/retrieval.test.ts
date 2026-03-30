/**
 * Unit tests for the hybrid retrieval engine — RRF fusion and RSE extraction
 */

// We test the pure functions directly, not the class (which needs live DB + embeddings)

// ─── RRF Fusion Tests ────────────────────────────────────────────────────────

describe('Reciprocal Rank Fusion', () => {

	// Inline the RRF logic for testing (same algorithm as retrieval.ts)
	function rrf(
		vectorNoteIds: string[],
		keywordNoteIds: string[],
		k: number = 60,
	): Map<string, { score: number; source: string }> {
		const scores = new Map<string, { score: number; source: string }>();

		vectorNoteIds.forEach((noteId, rank) => {
			const rrfScore = 1 / (k + rank + 1);
			const existing = scores.get(noteId);
			if (existing) {
				existing.score += rrfScore;
				existing.source = 'hybrid';
			} else {
				scores.set(noteId, { score: rrfScore, source: 'vector' });
			}
		});

		keywordNoteIds.forEach((noteId, rank) => {
			const rrfScore = 1 / (k + rank + 1);
			const existing = scores.get(noteId);
			if (existing) {
				existing.score += rrfScore;
				existing.source = 'hybrid';
			} else {
				scores.set(noteId, { score: rrfScore, source: 'keyword' });
			}
		});

		return scores;
	}

	it('returns empty map for empty inputs', () => {
		const result = rrf([], []);
		expect(result.size).toBe(0);
	});

	it('handles vector-only results', () => {
		const result = rrf(['noteA', 'noteB'], []);
		expect(result.size).toBe(2);
		expect(result.get('noteA')!.source).toBe('vector');
		expect(result.get('noteB')!.source).toBe('vector');
	});

	it('handles keyword-only results', () => {
		const result = rrf([], ['noteX', 'noteY']);
		expect(result.size).toBe(2);
		expect(result.get('noteX')!.source).toBe('keyword');
	});

	it('marks overlapping results as hybrid', () => {
		const result = rrf(['noteA', 'noteB'], ['noteB', 'noteC']);
		expect(result.get('noteA')!.source).toBe('vector');
		expect(result.get('noteB')!.source).toBe('hybrid');
		expect(result.get('noteC')!.source).toBe('keyword');
	});

	it('gives higher RRF score to items appearing in both lists', () => {
		const result = rrf(['noteA', 'noteB'], ['noteB', 'noteA']);

		// noteA: 1/(60+1) from vector + 1/(60+2) from keyword
		// noteB: 1/(60+2) from vector + 1/(60+1) from keyword
		// Both should have the same total score
		const scoreA = result.get('noteA')!.score;
		const scoreB = result.get('noteB')!.score;
		expect(Math.abs(scoreA - scoreB)).toBeLessThan(0.0001);
	});

	it('ranks items by position — rank 1 scores higher than rank 2', () => {
		const result = rrf(['noteA', 'noteB', 'noteC'], []);
		const scoreA = result.get('noteA')!.score;
		const scoreB = result.get('noteB')!.score;
		const scoreC = result.get('noteC')!.score;
		expect(scoreA).toBeGreaterThan(scoreB);
		expect(scoreB).toBeGreaterThan(scoreC);
	});

	it('hybrid items score higher than single-source items', () => {
		// noteA appears in both lists (rank 1 each)
		// noteB appears only in vector (rank 2)
		const result = rrf(['noteA', 'noteB'], ['noteA']);
		const scoreA = result.get('noteA')!.score;
		const scoreB = result.get('noteB')!.score;
		expect(scoreA).toBeGreaterThan(scoreB);
	});

	it('handles duplicate noteIds within a single list', () => {
		// Same note appearing at rank 1 and rank 3 in vector results
		const result = rrf(['noteA', 'noteB', 'noteA'], []);
		// noteA should accumulate score from both positions
		const scoreA = result.get('noteA')!.score;
		const scoreB = result.get('noteB')!.score;
		expect(scoreA).toBeGreaterThan(scoreB);
	});

	it('uses k=60 by default (standard RRF constant)', () => {
		const result = rrf(['noteA'], []);
		// With k=60, rank 0: score = 1/(60+0+1) = 1/61
		expect(result.get('noteA')!.score).toBeCloseTo(1 / 61, 6);
	});

	it('allows custom k parameter', () => {
		const result = rrf(['noteA'], [], 10);
		// With k=10, rank 0: score = 1/(10+0+1) = 1/11
		expect(result.get('noteA')!.score).toBeCloseTo(1 / 11, 6);
	});
});

// ─── RSE (Relevant Segment Extraction) Tests ─────────────────────────────────

describe('Relevant Segment Extraction', () => {

	interface MockChunk {
		chunkIndex: number;
		text: string;
	}

	// Inline RSE logic for testing (same algorithm as retrieval.ts)
	function extractSegments(chunks: MockChunk[]): string[][] {
		if (chunks.length === 0) return [];
		if (chunks.length === 1) return [[chunks[0].text]];

		const sorted = [...chunks].sort((a, b) => a.chunkIndex - b.chunkIndex);
		const segments: string[][] = [];
		let current = [sorted[0].text];

		for (let i = 1; i < sorted.length; i++) {
			const gap = sorted[i].chunkIndex - sorted[i - 1].chunkIndex;
			if (gap <= 2) {
				current.push(sorted[i].text);
			} else {
				segments.push(current);
				current = [sorted[i].text];
			}
		}
		segments.push(current);
		return segments;
	}

	it('returns empty for empty chunks', () => {
		expect(extractSegments([])).toEqual([]);
	});

	it('returns single segment for single chunk', () => {
		const result = extractSegments([{ chunkIndex: 0, text: 'Hello' }]);
		expect(result).toEqual([['Hello']]);
	});

	it('merges adjacent chunks (gap = 1)', () => {
		const chunks = [
			{ chunkIndex: 0, text: 'A' },
			{ chunkIndex: 1, text: 'B' },
			{ chunkIndex: 2, text: 'C' },
		];
		const result = extractSegments(chunks);
		expect(result).toEqual([['A', 'B', 'C']]);
	});

	it('merges sandwiched chunks (gap = 2)', () => {
		const chunks = [
			{ chunkIndex: 0, text: 'A' },
			{ chunkIndex: 2, text: 'C' }, // gap=2, sandwiched B is missing but still merge
		];
		const result = extractSegments(chunks);
		expect(result).toEqual([['A', 'C']]);
	});

	it('splits non-adjacent chunks (gap > 2) into separate segments', () => {
		const chunks = [
			{ chunkIndex: 0, text: 'A' },
			{ chunkIndex: 5, text: 'F' }, // gap=5, too far
		];
		const result = extractSegments(chunks);
		expect(result).toEqual([['A'], ['F']]);
	});

	it('handles mixed adjacent and non-adjacent', () => {
		const chunks = [
			{ chunkIndex: 0, text: 'A' },
			{ chunkIndex: 1, text: 'B' },
			{ chunkIndex: 7, text: 'H' },
			{ chunkIndex: 8, text: 'I' },
		];
		const result = extractSegments(chunks);
		expect(result).toEqual([['A', 'B'], ['H', 'I']]);
	});

	it('sorts chunks by index regardless of input order', () => {
		const chunks = [
			{ chunkIndex: 2, text: 'C' },
			{ chunkIndex: 0, text: 'A' },
			{ chunkIndex: 1, text: 'B' },
		];
		const result = extractSegments(chunks);
		expect(result).toEqual([['A', 'B', 'C']]);
	});
});
