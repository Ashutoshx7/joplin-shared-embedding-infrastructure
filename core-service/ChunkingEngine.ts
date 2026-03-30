/**
 * Markdown-aware chunking engine for Joplin notes.
 * Splits at heading boundaries, preserves code blocks.
 */

import { createHash } from 'crypto';

export interface NoteChunk {
	noteId: string;
	chunkIndex: number;
	headingPath: string;
	text: string;
	noteTitle: string;
	notebookId: string;
	contentHash: string;
}

export class ChunkingEngine {
	private maxTokens: number;
	private overlapWords: number;

	constructor(maxTokens = 350, overlapWords = 50) {
		this.maxTokens = maxTokens;
		this.overlapWords = overlapWords;
	}

	/**
	 * Split a note into heading-aligned chunks with breadcrumb prefixes.
	 */
	public chunkNote(noteId: string, noteTitle: string, body: string, notebookId = ''): NoteChunk[] {
		if (!body || !body.trim()) return [];

		const contentHash = this.computeHash(body);
		const sections = this.splitByHeadings(body);
		const chunks: NoteChunk[] = [];

		for (const section of sections) {
			const heading = section.heading || '';
			const text = this.stripMarkdown(section.content).trim();
			if (!text || this.wordCount(text) < 3) continue;

			const prefix = heading ? `${noteTitle} > ${heading}` : noteTitle;
			const fullText = `${prefix}\n${text}`;

			// Split long sections
			if (this.wordCount(fullText) > this.maxTokens) {
				const subChunks = this.splitLongText(fullText, this.maxTokens, this.overlapWords);
				for (const sub of subChunks) {
					chunks.push({
						noteId,
						chunkIndex: chunks.length,
						headingPath: heading,
						text: sub,
						noteTitle,
						notebookId,
						contentHash,
					});
				}
			} else {
				chunks.push({
					noteId,
					chunkIndex: chunks.length,
					headingPath: heading,
					text: fullText,
					noteTitle,
					notebookId,
					contentHash,
				});
			}
		}

		return chunks;
	}

	public computeHash(content: string): string {
		return createHash('sha256').update(content).digest('hex').substring(0, 16);
	}

	private splitByHeadings(body: string): { heading: string; content: string }[] {
		const lines = body.split('\n');
		const sections: { heading: string; content: string }[] = [];
		let currentHeading = '';
		let currentContent: string[] = [];
		let inCodeBlock = false;

		for (const line of lines) {
			if (line.trim().startsWith('```')) {
				inCodeBlock = !inCodeBlock;
				currentContent.push(line);
				continue;
			}
			if (inCodeBlock) {
				currentContent.push(line);
				continue;
			}

			const headingMatch = line.match(/^(#{1,6})\s+(.+)/);
			if (headingMatch) {
				// Save previous section if it has content
				if (currentContent.length > 0 || currentHeading) {
					sections.push({ heading: currentHeading, content: currentContent.join('\n') });
				}
				currentHeading = headingMatch[2].trim();
				currentContent = [];
			} else {
				currentContent.push(line);
			}
		}

		if (currentContent.length > 0 || currentHeading) {
			sections.push({ heading: currentHeading, content: currentContent.join('\n') });
		}

		return sections;
	}

	private stripMarkdown(text: string): string {
		return text
			.replace(/#{1,6}\s+/g, '')
			.replace(/\*\*(.+?)\*\*/g, '$1')
			.replace(/\*(.+?)\*/g, '$1')
			.replace(/`(.+?)`/g, '$1')
			.replace(/\[(.+?)\]\(.+?\)/g, '$1')
			.replace(/!\[.*?\]\(.+?\)/g, '')
			.replace(/^[-*+]\s+/gm, '')
			.replace(/^\d+\.\s+/gm, '')
			.replace(/^>\s+/gm, '')
			.trim();
	}

	private wordCount(text: string): number {
		return text.split(/\s+/).filter(w => w.length > 0).length;
	}

	private splitLongText(text: string, maxWords: number, overlap: number): string[] {
		const words = text.split(/\s+/).filter(w => w.length > 0);
		const chunks: string[] = [];
		let start = 0;

		while (start < words.length) {
			const end = Math.min(start + maxWords, words.length);
			chunks.push(words.slice(start, end).join(' '));
			start = end - overlap;
			if (start >= words.length || end === words.length) break;
		}

		return chunks;
	}
}
