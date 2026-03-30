/**
 * Sample Consumer Plugin: Related Notes Sidebar
 *
 * Demonstrates consuming the Shared Embedding & Retrieval Infrastructure API.
 * This plugin does NOT generate embeddings itself — it calls the shared
 * aiSearch.* commands exposed by the infrastructure plugin.
 *
 * Purpose: Prove that the API contract works for downstream consumers.
 * Per @laurent: "a sample plugin should also be included to demonstrate the API"
 *
 * Commands consumed:
 *   aiSearch.isReady()              → check if index is ready
 *   aiSearch.findSimilar(noteId, k) → get semantically similar notes
 *   aiSearch.getNoteEmbedding(id)   → get note-level embedding vector
 *   aiSearch.embed(text)            → embed arbitrary text
 *   aiSearch.query(text, options)   → hybrid search
 */

import joplin from 'api';
import { ToolbarButtonLocation } from 'api/types';

joplin.plugins.register({
	onStart: async function () {
		console.info('Related Notes: starting sample consumer plugin');

		// ─── Create Panel ─────────────────────────────────────────────
		const panel = await joplin.views.panels.create('relatedNotesPanel');
		await joplin.views.panels.addScript(panel, './panel.css');

		// Initial HTML
		await joplin.views.panels.setHtml(panel, `
			<div id="related-notes-container">
				<h3>🔗 Related Notes</h3>
				<p class="hint">Select a note to find related content…</p>
			</div>
		`);

		// ─── Panel Message Handler ────────────────────────────────────
		await joplin.views.panels.onMessage(panel, async (msg: any) => {
			if (msg.type === 'openNote' && msg.noteId) {
				await joplin.commands.execute('openNote', msg.noteId);
			}
		});

		// ─── Toggle Command ───────────────────────────────────────────
		await joplin.commands.register({
			name: 'relatedNotes.toggle',
			label: 'Toggle Related Notes Sidebar',
			iconName: 'fas fa-project-diagram',
			execute: async () => {
				const visible = await joplin.views.panels.visible(panel);
				await joplin.views.panels.show(panel, !visible);
			},
		});

		await joplin.views.toolbarButtons.create(
			'relatedNotesBtn',
			'relatedNotes.toggle',
			ToolbarButtonLocation.NoteToolbar,
		);

		// ─── Note Selection Listener ──────────────────────────────────
		async function onNoteChange() {
			try {
				// Step 1: Check if the embedding index is ready
				let isReady = false;
				try {
					isReady = await joplin.commands.execute('aiSearch.isReady');
				} catch {
					// Infrastructure plugin not installed
					await joplin.views.panels.setHtml(panel, `
						<div id="related-notes-container">
							<h3>🔗 Related Notes</h3>
							<p class="error">⚠️ AI Search infrastructure not available.<br/>
							Install the Shared Embedding & Retrieval plugin first.</p>
						</div>
					`);
					return;
				}

				if (!isReady) {
					await joplin.views.panels.setHtml(panel, `
						<div id="related-notes-container">
							<h3>🔗 Related Notes</h3>
							<p class="hint">Index not ready. Build the AI Search index first.</p>
						</div>
					`);
					return;
				}

				// Step 2: Get current note
				const note = await joplin.workspace.selectedNote();
				if (!note) return;

				// Step 3: Show loading state
				await joplin.views.panels.setHtml(panel, `
					<div id="related-notes-container">
						<h3>🔗 Related Notes</h3>
						<p class="hint">Finding related notes…</p>
					</div>
				`);

				// Step 4: Call the shared infrastructure API
				// This is the key demonstration — we call aiSearch.findSimilar
				// which is provided by the infrastructure, not by this plugin
				const similar = await joplin.commands.execute(
					'aiSearch.findSimilar', note.id, 5
				);

				// Step 5: Also demonstrate aiSearch.getNoteEmbedding
				const embedding = await joplin.commands.execute(
					'aiSearch.getNoteEmbedding', note.id
				);

				// Step 6: Render results
				if (!similar || similar.length === 0) {
					await joplin.views.panels.setHtml(panel, `
						<div id="related-notes-container">
							<h3>🔗 Related Notes</h3>
							<p class="hint">No related notes found for "${escapeHtml(note.title)}"</p>
							${embedding ? `<p class="meta">📐 Embedding: ${embedding.length}-dim vector</p>` : ''}
						</div>
					`);
					return;
				}

				const resultsHtml = similar.map((r: any) => `
					<div class="related-note" onclick="
						if(typeof webviewApi !== 'undefined')
							webviewApi.postMessage({ type: 'openNote', noteId: '${r.noteId}' });
					">
						<div class="note-title">📄 ${escapeHtml(r.noteTitle || r.noteId)}</div>
						<div class="note-score">
							<span class="score-badge ${r.score > 0.8 ? 'high' : r.score > 0.6 ? 'mid' : 'low'}">
								${(r.score * 100).toFixed(0)}%
							</span>
						</div>
					</div>
				`).join('');

				await joplin.views.panels.setHtml(panel, `
					<div id="related-notes-container">
						<h3>🔗 Related Notes</h3>
						<p class="meta">
							For: <strong>${escapeHtml(note.title)}</strong>
							${embedding ? ` · ${embedding.length}-dim embedding` : ''}
						</p>
						<div class="results">${resultsHtml}</div>
						<p class="footer">
							Powered by <code>aiSearch.findSimilar</code> ·
							<code>aiSearch.getNoteEmbedding</code>
						</p>
					</div>
				`);
			} catch (err) {
				console.error('Related Notes error:', err);
				await joplin.views.panels.setHtml(panel, `
					<div id="related-notes-container">
						<h3>🔗 Related Notes</h3>
						<p class="error">❌ ${escapeHtml(String(err))}</p>
					</div>
				`);
			}
		}

		// Listen for note selection changes
		await joplin.workspace.onNoteSelectionChange(onNoteChange);

		// Also run on startup
		setTimeout(onNoteChange, 2000);

		console.info('Related Notes: sample consumer plugin ready');
	},
});

function escapeHtml(str: string): string {
	return str
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;');
}
