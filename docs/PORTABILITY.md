# Moving the archive between devices

YouScoop saves a source into the desktop library in one action. Its `.sacri` export contains that source's preserved media, captions, frames, descriptions, and evidence references. SACRiNOMiCON can import this source package independently of YouScoop's directories.

SACRiNOMiCON's Library screen also exports a `.sacri-library` collection. This includes every preserved source revision, current and retired evidence, user notes and their recorded edit history, manual claims, concept anchors, typed relationships, and original artifacts. Models are installed separately. Oracle conversation history currently remains on its device and is not part of this library export.

On desktop, Export library creates the package and reveals it in Finder. On Android, it opens the system document picker to choose a destination. Import source or library accepts either package type. This is a file handoff; automatic device synchronization is still pending.

Imports verify artifact sizes and SHA-256 checksums before changing knowledge. These checks detect corruption; they do not authenticate a speaker's allegation. A corrupt file or reused immutable record ID with different content fails the import transaction.

Artifact IDs and source revision IDs include artifact content hashes. If a raw inspection output changes while the searchable evidence stays the same, the next save creates a separate preserved revision. Repeating an identical save is idempotent. A different source revision can be selected from the archive's revision viewer.

Local notes are retained. Different imported notes appear under Notes to reconcile. Keep this device's note or Use imported note selects the active assessment, while both versions remain preserved. Choosing an edited note through the evidence inspector also records its history. Reimporting the same resolved conflict does not raise it again.

The evidence inspector also offers a corrected reading of an extracted passage. The Oracle uses this explicitly labeled human reading while the original extraction remains visible and preserved. Changing a note or standing retains the corrected reading. Use original reading removes the active correction while retaining its edit history. Library exports include these readings; conflicting imported readings require the same review as notes. Existing libraries and older packages load with no corrected reading by default.

Spelling corrections for this source applies explicit whole-term, case-sensitive substitutions to that source's Oracle readings. It preserves the originals and does not affect another source. Use a specific phrase if a word has several meanings. Substitutions use one pass and do not cascade into each other. The inspector displays the current Oracle reading alongside the original extraction. Source spellings are also included in library history, export and conflict review.

Reprocessing can advance a source while retaining its older evidence and assessments. Importing an older snapshot keeps the newer local revision. Divergent revisions appear under Source revisions to review. Their passages remain preserved, and the local revision remains active until you choose another in the source's Preserved revisions section. Selecting a revision changes active retrieval without deleting the other revisions or their annotations.

The current portable format caps the complete archive package at 8 GiB, an individual artifact at 4 GiB, and the typed library manifest at 128 MiB. Export/import streams artifact bytes rather than encoding a whole video in frontend memory. Large-library memory and phone latency remain unmeasured.

## Verified behavior

The earlier library snapshot was exported and restored into a separate directory with 112 evidence passages, 67 checksummed artifacts, both source assessments, and the Baal correction. An intermediate package containing twelve unreviewed Omni descriptions had 124 evidence passages and 69 artifacts. After observing invented model details, the current export excludes raw Omni descriptions from evidence while retaining their raw output file as an artifact. The current test source exports 112 evidence passages and 69 artifacts; a human-written, reviewed observation can add a new passage. An ARM64 Android emulator storage probe imported, reimported, exported, and restored this current package with 112 current evidence passages and no duplicate on repeat import. A native isolated test confirmed raw fabricated text was absent from search and export, while the reviewed observation appeared in both. Separate preservation tests cover manual graph records, retired evidence, edit history, note conflicts, revision conflicts, older snapshots, transactional rollback, and corruption.

Android release packaging, emulator startup and a native WebView-to-Tauri import, correction and search integration test pass. The Android document picker interaction and physical phone import still require runtime verification.
