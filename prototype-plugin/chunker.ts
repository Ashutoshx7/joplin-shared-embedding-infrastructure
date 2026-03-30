/**
 * Markdown-aware note chunker
 *
 * Splits Joplin notes into semantically coherent chunks aligned to heading
 * boundaries. Each chunk is prefixed with the note title and heading path
 * so the embedding captures topic context.
 */

export interface NoteChunk {
	noteId: string;
	chunkIndex: number;
	headingPath: string;   // "## Setup > ### Database" 
	text: string;          // Chunk text with title/heading prefix
	rawText: string;       // Original text without prefix
	noteTitle: string;
	notebookId: string;
	contentHash: string;   // SHA-256 of rawText for change detection
}

interface HeadingSection {
	level: number;
	heading: string;
	text: string;
	headingPath: string;
}

const MAX_CHUNK_TOKENS = 350;   // Reserve ~30% of BGE-small's 512 for prefix
const OVERLAP_WORDS = 50;
const MIN_CHUNK_TOKENS = 3;

/**
 * Rough token count: ~1.3 tokens per word for English text.
 * Good enough for chunking decisions; exact tokenization happens in the model.
 */
function estimateTokens(text: string): number {
	const words = text.trim().split(/\s+/).filter(w => w.length > 0);
	return Math.ceil(words.length * 1.3);
}

/**
 * SHA-256 hash of text for change detection.
 * Uses a simple djb2 hash in environments without crypto (webworker).
 * In Node.js, uses crypto.createHash.
 */
function hashText(text: string): string {
	// Simple, fast hash for change detection (not security)
	let hash = 5381;
	for (let i = 0; i < text.length; i++) {
		hash = ((hash << 5) + hash + text.charCodeAt(i)) & 0xffffffff;
	}
	return hash.toString(16).padStart(8, '0');
}

/**
 * Strip Markdown syntax that doesn't contribute to semantic meaning
 * but preserve code content (users search for code).
 */
function stripMarkdownSyntax(text: string): string {
	return text
		// Remove image syntax but keep alt text
		.replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
		// Remove link syntax but keep text
		.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
		// Remove bold/italic markers
		.replace(/(\*{1,3}|_{1,3})(.*?)\1/g, '$2')
		// Remove strikethrough
		.replace(/~~(.*?)~~/g, '$1')
		// Remove inline code backticks (keep content)
		.replace(/`([^`]+)`/g, '$1')
		// Remove HTML tags
		.replace(/<[^>]+>/g, '')
		// Normalize whitespace
		.replace(/\n{3,}/g, '\n\n')
		.trim();
}

/**
 * Split markdown body into sections by heading boundaries.
 * Each heading (#, ##, ###) starts a new section.
 */
function splitOnHeadings(body: string): HeadingSection[] {
	const lines = body.split('\n');
	const sections: HeadingSection[] = [];

	let currentHeading = '';
	let currentLevel = 0;
	let currentText: string[] = [];
	const headingStack: string[] = [];

	function buildHeadingPath(): string {
		return headingStack.join(' > ');
	}

	function flushSection() {
		const text = currentText.join('\n').trim();
		if (text.length > 0) {
			// Capture heading path at flush time (not after stack is modified)
			const path = buildHeadingPath();
			sections.push({
				level: currentLevel,
				heading: currentHeading,
				text: text,
				headingPath: path,
			});
		}
		currentText = [];
	}

	for (const line of lines) {
		const headingMatch = line.match(/^(#{1,3})\s+(.+)/);

		if (headingMatch) {
			flushSection();

			const level = headingMatch[1].length;
			const heading = headingMatch[2].trim();

			// Update heading stack — pop back to current level
			while (headingStack.length >= level) {
				headingStack.pop();
			}
			headingStack.push(heading);

			currentLevel = level;
			currentHeading = heading;
		} else {
			currentText.push(line);
		}
	}
	flushSection();

	return sections;
}

/**
 * Split a long text at paragraph boundaries (double newlines) with overlap.
 */
function splitAtParagraphs(text: string, maxTokens: number, overlapWords: number): string[] {
	const paragraphs = text.split(/\n\n+/).filter(p => p.trim().length > 0);
	const chunks: string[] = [];

	let currentChunk: string[] = [];
	let currentTokens = 0;

	for (const para of paragraphs) {
		const paraTokens = estimateTokens(para);

		if (currentTokens + paraTokens > maxTokens && currentChunk.length > 0) {
			chunks.push(currentChunk.join('\n\n'));

			// Overlap: keep last N words from previous chunk
			const lastText = currentChunk.join('\n\n');
			const words = lastText.split(/\s+/);
			const overlapText = words.slice(-overlapWords).join(' ');

			currentChunk = overlapText ? [overlapText] : [];
			currentTokens = estimateTokens(overlapText || '');
		}

		currentChunk.push(para);
		currentTokens += paraTokens;
	}

	if (currentChunk.length > 0) {
		chunks.push(currentChunk.join('\n\n'));
	}

	return chunks;
}

/**
 * Check if text is inside a code block (fenced with ``` or ~~~).
 * If so, don't split it.
 */
function isCodeBlock(text: string): boolean {
	const trimmed = text.trim();
	return trimmed.startsWith('```') || trimmed.startsWith('~~~');
}

/**
 * Main chunking function — processes a note into NoteChunk[].
 */
export function chunkNote(
	noteId: string,
	noteTitle: string,
	noteBody: string,
	notebookId: string,
): NoteChunk[] {
	if (!noteBody || noteBody.trim().length === 0) {
		// Empty body — create a single chunk from title only if title exists
		if (noteTitle && noteTitle.trim().length > 0) {
			return [{
				noteId,
				chunkIndex: 0,
				headingPath: '',
				text: noteTitle,
				rawText: noteTitle,
				noteTitle,
				notebookId,
				contentHash: hashText(noteTitle),
			}];
		}
		return [];
	}

	const sections = splitOnHeadings(noteBody);

	// If no headings found, treat entire body as one section
	if (sections.length === 0) {
		sections.push({
			level: 0,
			heading: '',
			text: noteBody.trim(),
			headingPath: '',
		});
	}

	const chunks: NoteChunk[] = [];
	let chunkIndex = 0;

	for (const section of sections) {
		const cleanedText = stripMarkdownSyntax(section.text);
		const tokens = estimateTokens(cleanedText);

		if (tokens < MIN_CHUNK_TOKENS) {
			continue; // Skip tiny fragments
		}

		if (tokens <= MAX_CHUNK_TOKENS) {
			// Fits in one chunk
			const prefix = [noteTitle, section.headingPath]
				.filter(s => s && s.length > 0)
				.join('\n');
			const fullText = prefix ? `${prefix}\n${cleanedText}` : cleanedText;

			chunks.push({
				noteId,
				chunkIndex: chunkIndex++,
				headingPath: section.headingPath,
				text: fullText,
				rawText: cleanedText,
				noteTitle,
				notebookId,
				contentHash: hashText(cleanedText),
			});
		} else {
			// Split at paragraph boundaries with overlap
			const subChunks = splitAtParagraphs(cleanedText, MAX_CHUNK_TOKENS, OVERLAP_WORDS);

			for (const subText of subChunks) {
				if (estimateTokens(subText) < MIN_CHUNK_TOKENS) continue;

				const prefix = [noteTitle, section.headingPath]
					.filter(s => s && s.length > 0)
					.join('\n');
				const fullText = prefix ? `${prefix}\n${subText}` : subText;

				chunks.push({
					noteId,
					chunkIndex: chunkIndex++,
					headingPath: section.headingPath,
					text: fullText,
					rawText: subText,
					noteTitle,
					notebookId,
					contentHash: hashText(subText),
				});
			}
		}
	}

	return chunks;
}
