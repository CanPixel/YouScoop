use knowledge_core::*;
use std::io::Write;

fn fixture() -> SourcePackage {
    let provenance = Provenance {
        method: "captions".into(),
        model: None,
        generated_at: 1,
        original_asset: None,
    };
    SourcePackage {
        schema_version: SCHEMA_VERSION,
        revision: "revision-a".into(),
        exported_at: 1,
        source: Source {
            id: "source-a".into(),
            title: "A recorded claim".into(),
            uri: "https://example.org/video".into(),
            publisher: "Speaker".into(),
            duration_seconds: 90.0,
            captured_at: 1,
            summary: None,
        },
        embedding_model: EMBEDDING_MODEL.into(),
        evidence: vec![Evidence {
            id: "evidence-a".into(),
            source_id: "source-a".into(),
            start: 20.0,
            end: 30.0,
            text: "The speaker discusses symbolic egregores.".into(),
            kind: EvidenceKind::Speech,
            provenance: provenance.clone(),
            vector: vec![0.0; EMBEDDING_DIM],
        }],
        claims: vec![Claim {
            id: "claim-a".into(),
            source_id: "source-a".into(),
            statement: "Collective attention can create an egregore.".into(),
            kind: "metaphysical_hypothesis".into(),
            evidence_ids: vec!["evidence-a".into()],
            provenance,
        }],
        concepts: vec![Concept {
            id: "concept-a".into(),
            label: "Egregores".into(),
            aliases: vec![],
            evidence_ids: vec!["evidence-a".into()],
        }],
        relationships: vec![],
        assets: vec![],
    }
}

#[test]
fn corrected_readings_are_searchable_reversible_and_preserve_originals_and_notes() {
    let dir = tempfile::tempdir().unwrap();
    let mut archive = Archive::open(dir.path()).unwrap();
    let mut p = fixture();
    p.evidence[0].text = "The speaker calls the account Ball.".into();
    archive.import(&p, dir.path()).unwrap();
    archive
        .assess("evidence-a", "uncertain", "An attributed allegation.")
        .unwrap();
    let corrected = "The speaker calls the account Baal.";
    let saved = archive
        .correct_reading("evidence-a", Some(corrected))
        .unwrap();
    assert_eq!(saved.status, "uncertain");
    assert_eq!(saved.note, "An attributed allegation.");
    assert_eq!(
        archive.evidence("evidence-a").unwrap().text,
        p.evidence[0].text
    );
    let hits = archive.search("Baal", None, None, 6).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0]
            .assessment
            .as_ref()
            .unwrap()
            .corrected_text
            .as_deref(),
        Some(corrected)
    );
    let updated = archive
        .assess(
            "evidence-a",
            "contested",
            "Preserve the reading when changing my note.",
        )
        .unwrap();
    assert_eq!(updated.corrected_text.as_deref(), Some(corrected));
    assert!(archive
        .correct_reading("source-a", Some("Invalid target"))
        .is_err());
    assert!(archive.correct_reading("evidence-a", Some(" ")).is_err());
    archive.correct_reading("evidence-a", None).unwrap();
    assert!(archive.search("Baal", None, None, 6).unwrap().is_empty());
    assert!(archive
        .snapshot()
        .unwrap()
        .assessment_history
        .iter()
        .any(|a| a.corrected_text.as_deref() == Some(corrected)));
}

#[test]
fn corrected_readings_survive_backup_and_differences_require_review() {
    let dir = tempfile::tempdir().unwrap();
    let mut desktop = Archive::open(dir.path().join("desktop")).unwrap();
    desktop.import(&fixture(), dir.path()).unwrap();
    desktop
        .correct_reading("evidence-a", Some("Desktop's corrected reading."))
        .unwrap();
    let file = dir.path().join("correction.sacri-library");
    desktop.export_library(&file).unwrap();
    let mut phone = Archive::open(dir.path().join("phone")).unwrap();
    phone.import_portable(&file).unwrap();
    assert_eq!(
        phone.summary().unwrap().assessments[0]
            .corrected_text
            .as_deref(),
        Some("Desktop's corrected reading.")
    );
    phone
        .correct_reading("evidence-a", Some("Phone's corrected reading."))
        .unwrap();
    assert_eq!(
        phone.import_portable(&file).unwrap().annotation_conflicts,
        1
    );
    assert_eq!(
        phone.summary().unwrap().assessments[0]
            .corrected_text
            .as_deref(),
        Some("Phone's corrected reading.")
    );
    let conflict = phone.annotation_conflicts().unwrap().remove(0);
    phone.resolve_annotation(&conflict.id, true).unwrap();
    assert_eq!(
        phone.summary().unwrap().assessments[0]
            .corrected_text
            .as_deref(),
        Some("Desktop's corrected reading.")
    );
    assert_eq!(
        phone.import_portable(&file).unwrap().annotation_conflicts,
        0
    );
    assert_eq!(
        phone.evidence("evidence-a").unwrap().text,
        fixture().evidence[0].text
    );
    let mut invalid = phone.snapshot().unwrap();
    invalid.assessments[0].entity_id = "source-a".into();
    assert!(invalid.validate().is_err());
}

#[test]
fn legacy_assessments_migrate_without_changing_serialized_history_identity() {
    let dir = tempfile::tempdir().unwrap();
    let db = rusqlite::Connection::open(dir.path().join("archive.sqlite")).unwrap();
    db.execute_batch("CREATE TABLE assessments(entity_id TEXT PRIMARY KEY,status TEXT NOT NULL,note TEXT NOT NULL,updated_at INTEGER NOT NULL);
        INSERT INTO assessments VALUES('evidence-a','uncertain','Legacy annotation',1);").unwrap();
    drop(db);
    let mut archive = Archive::open(dir.path()).unwrap();
    archive.import(&fixture(), dir.path()).unwrap();
    let legacy = archive.summary().unwrap().assessments.remove(0);
    assert_eq!(
        serde_json::to_string(&legacy).unwrap(),
        r#"{"entityId":"evidence-a","status":"uncertain","note":"Legacy annotation","updatedAt":1}"#
    );
    archive
        .correct_reading("evidence-a", Some("Human correction."))
        .unwrap();
    assert!(archive
        .snapshot()
        .unwrap()
        .assessment_history
        .contains(&legacy));
    drop(archive);
    assert_eq!(
        Archive::open(dir.path())
            .unwrap()
            .summary()
            .unwrap()
            .assessments[0]
            .corrected_text
            .as_deref(),
        Some("Human correction.")
    );
}

#[test]
fn source_spellings_apply_with_boundaries_scope_history_and_conflict_review() {
    let dir = tempfile::tempdir().unwrap();
    let mut archive = Archive::open(dir.path().join("desktop")).unwrap();
    let mut p = fixture();
    p.evidence[0].text = "Ball, Ballarat and a ball; Baal.".into();
    archive.import(&p, dir.path()).unwrap();
    let mut other = p.clone();
    other.source.id = "other".into();
    other.evidence[0].id = "other-evidence".into();
    other.evidence[0].source_id = "other".into();
    other.claims.clear();
    other.concepts.clear();
    archive.import(&other, dir.path()).unwrap();
    let terms = vec![
        SpellingCorrection {
            original: "Ball".into(),
            corrected: "Baal".into(),
        },
        SpellingCorrection {
            original: "Baal".into(),
            corrected: "Baal deity".into(),
        },
    ];
    archive
        .assess("source-a", "collected", "Keep this source note.")
        .unwrap();
    archive
        .correct_source_spellings("source-a", &terms)
        .unwrap();
    let hits = archive.search("Baal", None, Some("source-a"), 6).unwrap();
    assert_eq!(
        hits[0].reading.as_deref(),
        Some("Baal, Ballarat and a ball; Baal deity.")
    );
    assert_eq!(hits[0].evidence.text, p.evidence[0].text);
    assert_eq!(
        archive.reading("evidence-a").unwrap(),
        "Baal, Ballarat and a ball; Baal deity."
    );
    assert_eq!(
        archive.search("Baal", None, Some("other"), 6).unwrap()[0].reading,
        None
    );
    // Correction-only terms participate in lexical lookup, including Unicode boundaries.
    let terms = vec![SpellingCorrection {
        original: "Ball".into(),
        corrected: "Bāal".into(),
    }];
    archive
        .correct_source_spellings("source-a", &terms)
        .unwrap();
    assert_eq!(
        archive.search("Bāal", None, None, 6).unwrap()[0]
            .evidence
            .id,
        "evidence-a"
    );
    assert_eq!(
        corrected_spelling("çBall Ballç Ball", &terms),
        "çBall Ballç Bāal"
    );
    let file = dir.path().join("source-spellings.sacri-library");
    archive.export_library(&file).unwrap();
    let mut phone = Archive::open(dir.path().join("phone")).unwrap();
    phone.import_portable(&file).unwrap();
    assert_eq!(
        phone.search("Bāal", None, None, 6).unwrap()[0]
            .reading
            .as_deref(),
        Some("Bāal, Ballarat and a ball; Baal.")
    );
    phone.correct_source_spellings("source-a", &[]).unwrap();
    assert_eq!(
        phone.import_portable(&file).unwrap().annotation_conflicts,
        1
    );
    let conflict = phone.annotation_conflicts().unwrap().remove(0);
    phone.resolve_annotation(&conflict.id, true).unwrap();
    let source = phone
        .summary()
        .unwrap()
        .assessments
        .into_iter()
        .find(|a| a.entity_id == "source-a")
        .unwrap();
    assert_eq!(source.source_spellings, terms);
    assert_eq!(source.note, "Keep this source note.");
    assert!(phone
        .correct_source_spellings("evidence-a", &terms)
        .is_err());
    let duplicate = vec![terms[0].clone(), terms[0].clone()];
    assert!(phone
        .correct_source_spellings("source-a", &duplicate)
        .is_err());
    assert_eq!(
        corrected_spelling(
            "Unchanged",
            &[SpellingCorrection {
                original: "".into(),
                corrected: "Never loop".into()
            }]
        ),
        "Unchanged"
    );
}

#[test]
fn repeated_import_and_reprocessing_preserve_assessments_and_history() {
    let dir = tempfile::tempdir().unwrap();
    let mut archive = Archive::open(dir.path()).unwrap();
    let mut package = fixture();
    archive.import(&package, dir.path()).unwrap();
    archive
        .assess(
            "claim-a",
            "uncertain",
            "A hypothesis, not independently demonstrated.",
        )
        .unwrap();
    package.exported_at = 200;
    archive.import(&package, dir.path()).unwrap();
    assert_eq!(archive.summary().unwrap().evidence_count, 1);
    package.revision = "revision-b".into();
    package.evidence[0].id = "evidence-b".into();
    package.evidence[0].text = "A corrected description of collective attention.".into();
    package.claims.clear();
    package.concepts[0].evidence_ids = vec!["evidence-b".into()];
    archive.import(&package, dir.path()).unwrap();
    let summary = archive.summary().unwrap();
    assert_eq!(summary.evidence_count, 1);
    assert_eq!(summary.assessments[0].status, "uncertain");
    assert_eq!(summary.claims.len(), 1);
    assert_eq!(summary.concepts[0].evidence_ids.len(), 2);
    assert!(archive
        .search("egregores", None, None, 10)
        .unwrap()
        .is_empty());
    assert_eq!(
        archive.search("corrected", None, None, 10).unwrap().len(),
        1
    );
}

#[test]
fn conflicting_revision_or_evidence_rolls_back_source_changes() {
    let dir = tempfile::tempdir().unwrap();
    let mut archive = Archive::open(dir.path()).unwrap();
    let mut package = fixture();
    archive.import(&package, dir.path()).unwrap();
    package.source.title = "Attempted overwrite".into();
    assert!(archive.import(&package, dir.path()).is_err());
    package.revision = "revision-b".into();
    package.evidence[0].text = "Reused ID, different evidence".into();
    assert!(archive.import(&package, dir.path()).is_err());
    assert_eq!(
        archive.source_detail("source-a").unwrap().source.title,
        "A recorded claim"
    );
    assert_eq!(archive.summary().unwrap().evidence_count, 1);
}

#[test]
fn portable_artifacts_survive_loss_of_original_directory() {
    let original = tempfile::tempdir().unwrap();
    let destination = tempfile::tempdir().unwrap();
    let file = original.path().join("captions.json");
    std::fs::write(&file, b"preserved timestamped speech").unwrap();
    let output = destination.path().join("source.sacri");
    let mut package = fixture();
    package.evidence[0].provenance.original_asset = Some("captions".into());
    write_package(
        package,
        &[AssetInput {
            id: "captions".into(),
            name: "captions.json".into(),
            kind: "captions".into(),
            path: file,
        }],
        &output,
    )
    .unwrap();
    drop(original);
    let mut archive = Archive::open(destination.path().join("library")).unwrap();
    archive.import_file(&output).unwrap();
    assert_eq!(
        std::fs::read(archive.asset_path("source-a", "captions").unwrap()).unwrap(),
        b"preserved timestamped speech"
    );
}

#[test]
fn corrupt_artifacts_cannot_change_archive() {
    let dir = tempfile::tempdir().unwrap();
    let mut package = fixture();
    package.assets.push(Asset {
        id: "captions".into(),
        name: "captions.json".into(),
        kind: "captions".into(),
        sha256: "0".repeat(64),
        bytes: 4,
    });
    let output = dir.path().join("corrupt.sacri");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&output).unwrap());
    let options = zip::write::SimpleFileOptions::default();
    zip.start_file("manifest.json", options).unwrap();
    zip.write_all(&serde_json::to_vec(&package).unwrap())
        .unwrap();
    zip.start_file("assets/captions.json", options).unwrap();
    zip.write_all(b"lies").unwrap();
    zip.finish().unwrap();
    let mut archive = Archive::open(dir.path().join("library")).unwrap();
    assert!(archive.import_file(&output).is_err());
    assert!(archive.summary().unwrap().sources.is_empty());
}

#[test]
fn invalid_references_paths_and_embeddings_are_rejected() {
    let mut package = fixture();
    package.evidence[0].vector = vec![1.0; 3];
    assert!(package.validate().is_err());
    package.evidence[0].vector.clear();
    package.claims[0].evidence_ids = vec!["missing".into()];
    assert!(package.validate().is_err());
    package.claims.clear();
    package.assets.push(Asset {
        id: "escape".into(),
        name: "../escape".into(),
        kind: "file".into(),
        sha256: "0".repeat(64),
        bytes: 1,
    });
    assert!(package.validate().is_err());
}

#[test]
fn hybrid_search_finds_semantic_evidence_and_respects_scope() {
    let dir = tempfile::tempdir().unwrap();
    let mut archive = Archive::open(dir.path()).unwrap();
    let mut package = fixture();
    package.evidence[0].vector[0] = 1.0;
    archive.import(&package, dir.path()).unwrap();
    let mut query = vec![0.0; EMBEDDING_DIM];
    query[0] = 1.0;
    assert_eq!(
        archive.search("unrelated", Some(&query), None, 6).unwrap()[0]
            .evidence
            .id,
        "evidence-a"
    );
    assert!(archive
        .search("egregores", Some(&query), Some("other-source"), 6)
        .unwrap()
        .is_empty());
}

#[test]
fn transcription_corrections_are_searchable_without_rewriting_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let mut archive = Archive::open(dir.path()).unwrap();
    let mut package = fixture();
    package.evidence[0].text = "The caption reads Ball, a phonetic rendering.".into();
    archive.import(&package, dir.path()).unwrap();
    archive.assess("evidence-a","uncertain","Transcription correction: the speaker says Baal, not Ball. This does not verify the allegation.").unwrap();
    let hits = archive
        .search("What does the speaker claim about Baal?", None, None, 6)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].evidence.text.contains("Ball"));
    assert!(!hits[0].evidence.text.contains("Baal"));
    assert!(hits[0].assessment.as_ref().unwrap().note.contains("Baal"));
}

#[test]
fn older_annotations_are_indexed_on_open_and_updates_are_atomic() {
    let dir = tempfile::tempdir().unwrap();
    let mut archive = Archive::open(dir.path()).unwrap();
    archive.import(&fixture(), dir.path()).unwrap();
    archive
        .assess("evidence-a", "uncertain", "Corrected spelling: Baal.")
        .unwrap();
    drop(archive);
    let conn = rusqlite::Connection::open(dir.path().join("archive.sqlite")).unwrap();
    conn.execute("DELETE FROM assessment_fts", []).unwrap();
    drop(conn);
    let archive = Archive::open(dir.path()).unwrap();
    assert_eq!(archive.search("Baal", None, None, 6).unwrap().len(), 1);
    archive
        .assess("evidence-a", "contested", "New interpretation: Asherah.")
        .unwrap();
    assert!(archive.search("Baal", None, None, 6).unwrap().is_empty());
    assert_eq!(archive.search("Asherah", None, None, 6).unwrap().len(), 1);
    let conn = rusqlite::Connection::open(dir.path().join("archive.sqlite")).unwrap();
    conn.execute("DROP TABLE assessment_fts", []).unwrap();
    assert!(archive
        .assess(
            "evidence-a",
            "supported",
            "An update that cannot be indexed."
        )
        .is_err());
    let saved = archive.summary().unwrap().assessments.remove(0);
    assert_eq!(saved.status, "contested");
    assert!(saved.note.contains("Asherah"));
}

#[test]
fn search_limits_apply_after_current_revision_and_source_scope() {
    let dir = tempfile::tempdir().unwrap();
    let mut archive = Archive::open(dir.path()).unwrap();
    let mut noisy = fixture();
    noisy.evidence = (0..350)
        .map(|i| {
            let mut e = noisy.evidence[0].clone();
            e.id = format!("noise-{i}");
            e.text = "Baal Baal Baal Baal Baal".into();
            e
        })
        .collect();
    noisy.claims.clear();
    noisy.concepts.clear();
    archive.import(&noisy, dir.path()).unwrap();
    let mut selected = fixture();
    selected.source.id = "selected-source".into();
    selected.revision = "selected-revision".into();
    selected.evidence[0].source_id = selected.source.id.clone();
    selected.evidence[0].id = "selected-evidence".into();
    selected.evidence[0].text = "Baal is mentioned.".into();
    selected.claims.clear();
    selected.concepts.clear();
    archive.import(&selected, dir.path()).unwrap();
    assert_eq!(
        archive
            .search("Baal", None, Some("selected-source"), 6)
            .unwrap()[0]
            .evidence
            .id,
        "selected-evidence"
    );
    noisy.revision = "noise-retired".into();
    noisy.evidence.clear();
    archive.import(&noisy, dir.path()).unwrap();
    assert_eq!(
        archive.search("Baal", None, None, 6).unwrap()[0]
            .evidence
            .id,
        "selected-evidence"
    );
}

#[test]
fn library_roundtrip_keeps_artifacts_retired_evidence_manual_graph_and_note_history() {
    let original = tempfile::tempdir().unwrap();
    let output = tempfile::tempdir().unwrap();
    let mut archive = Archive::open(original.path().join("library")).unwrap();
    let asset = original.path().join("captions.json");
    std::fs::write(&asset, b"original captions").unwrap();
    let mut p = fixture();
    p.evidence[0].provenance.original_asset = Some("captions".into());
    let p = write_package(
        p,
        &[AssetInput {
            id: "captions".into(),
            name: "captions.json".into(),
            kind: "captions".into(),
            path: asset,
        }],
        &original.path().join("source.sacri"),
    )
    .unwrap();
    archive
        .import_file(&original.path().join("source.sacri"))
        .unwrap();
    archive
        .assess("evidence-a", "uncertain", "First note: Ball.")
        .unwrap();
    archive
        .assess("evidence-a", "contested", "Correction: Baal.")
        .unwrap();
    let manual = archive
        .add_claim("evidence-a", "The spelling is Baal.", "interpretation")
        .unwrap();
    let concept = archive.add_concept("Baal", "evidence-a").unwrap();
    archive
        .add_relationship(
            "concept-a",
            &concept.id,
            RelationKind::Related,
            "evidence-a",
            "A shared symbolic context.",
        )
        .unwrap();
    let mut next = p.clone();
    next.revision = "revision-b".into();
    next.evidence[0].id = "evidence-b".into();
    next.evidence[0].text = "A new current passage about Baal.".into();
    next.claims.clear();
    next.concepts.clear();
    // Import the new revision using the verified preserved artifact.
    let asset_dir = original.path().join("unpacked");
    std::fs::create_dir(&asset_dir).unwrap();
    std::fs::write(asset_dir.join("captions.json"), b"original captions").unwrap();
    archive.import(&next, &asset_dir).unwrap();
    let backup = output.path().join("library.sacri-library");
    archive.export_library(&backup).unwrap();
    drop(archive);
    drop(original);
    let mut restored = Archive::open(output.path().join("restored")).unwrap();
    let result = restored.import_portable(&backup).unwrap();
    assert_eq!(result.annotation_conflicts, 0);
    restored.import_portable(&backup).unwrap();
    let summary = restored.summary().unwrap();
    assert_eq!(summary.evidence_count, 1);
    assert_eq!(summary.claims.len(), 2);
    assert_eq!(summary.relationships.len(), 1);
    assert!(summary.claims.iter().any(|c| c.id == manual.id));
    assert_eq!(summary.assessments[0].note, "Correction: Baal.");
    assert_eq!(
        restored.evidence("evidence-a").unwrap().text,
        "The speaker discusses symbolic egregores."
    );
    assert_eq!(
        std::fs::read(restored.asset_path("source-a", "captions").unwrap()).unwrap(),
        b"original captions"
    );
    let snapshot = restored.snapshot().unwrap();
    assert_eq!(snapshot.revisions.len(), 2);
    assert_eq!(snapshot.assessment_history.len(), 2);
    assert_eq!(snapshot.heads["source-a"], "revision-b");
}

#[test]
fn library_merge_keeps_local_notes_and_remembers_conflict_resolution() {
    let dir = tempfile::tempdir().unwrap();
    let mut desktop = Archive::open(dir.path().join("desktop")).unwrap();
    desktop.import(&fixture(), dir.path()).unwrap();
    desktop
        .assess("evidence-a", "uncertain", "Speaker means Baal.")
        .unwrap();
    let backup = dir.path().join("library.sacri-library");
    desktop.export_library(&backup).unwrap();
    let mut phone = Archive::open(dir.path().join("phone")).unwrap();
    phone.import_portable(&backup).unwrap();
    phone
        .assess(
            "evidence-a",
            "contested",
            "My phone note with further questions.",
        )
        .unwrap();
    let report = phone.import_portable(&backup).unwrap();
    assert_eq!(report.annotation_conflicts, 1);
    assert_eq!(
        phone.summary().unwrap().assessments[0].note,
        "My phone note with further questions."
    );
    let conflict = phone.annotation_conflicts().unwrap().remove(0);
    assert_eq!(conflict.incoming.note, "Speaker means Baal.");
    phone.resolve_annotation(&conflict.id, false).unwrap();
    assert_eq!(
        phone.import_portable(&backup).unwrap().annotation_conflicts,
        0
    );
    assert_eq!(phone.snapshot().unwrap().assessment_history.len(), 2);
    let second = dir.path().join("phone.sacri-library");
    phone.export_library(&second).unwrap();
    let mut other = Archive::open(dir.path().join("other")).unwrap();
    other.import_portable(&second).unwrap();
    assert_eq!(
        other.import_portable(&backup).unwrap().annotation_conflicts,
        0
    );
    assert_eq!(
        other.summary().unwrap().assessments[0].note,
        "My phone note with further questions."
    );
}

#[test]
fn divergent_source_revisions_are_preserved_without_switching_the_local_head() {
    let dir = tempfile::tempdir().unwrap();
    let mut local = Archive::open(dir.path().join("local")).unwrap();
    let mut remote = Archive::open(dir.path().join("remote")).unwrap();
    let mut p = fixture();
    local.import(&p, dir.path()).unwrap();
    remote.import(&p, dir.path()).unwrap();
    p.claims.clear();
    p.concepts.clear();
    p.revision = "revision-b".into();
    p.evidence[0].id = "evidence-b".into();
    p.evidence[0].text = "Local correction.".into();
    local.import(&p, dir.path()).unwrap();
    p.revision = "revision-c".into();
    p.evidence[0].id = "evidence-c".into();
    p.evidence[0].text = "Remote correction.".into();
    remote.import(&p, dir.path()).unwrap();
    let file = dir.path().join("remote.sacri-library");
    remote.export_library(&file).unwrap();
    let report = local.import_portable(&file).unwrap();
    assert_eq!(report.source_conflicts, vec!["source-a"]);
    let snapshot = local.snapshot().unwrap();
    assert_eq!(snapshot.heads["source-a"], "revision-b");
    assert_eq!(snapshot.revisions.len(), 3);
    assert_eq!(
        local.search("correction", None, None, 6).unwrap()[0]
            .evidence
            .id,
        "evidence-b"
    );
    assert_eq!(
        local.evidence("evidence-c").unwrap().text,
        "Remote correction."
    );
    assert_eq!(
        local.import_portable(&file).unwrap().source_conflicts,
        vec!["source-a"]
    );
    local.activate_revision("source-a", "revision-c").unwrap();
    assert!(local.source_conflicts().unwrap().is_empty());
    assert_eq!(local.snapshot().unwrap().heads["source-a"], "revision-c");
    assert_eq!(
        local.search("correction", None, None, 6).unwrap()[0]
            .evidence
            .id,
        "evidence-c"
    );
    assert_eq!(
        local.evidence("evidence-b").unwrap().text,
        "Local correction."
    );
}

#[test]
fn backup_import_advances_a_descendant_and_does_not_downgrade_to_an_older_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let mut desktop = Archive::open(dir.path().join("desktop")).unwrap();
    let mut p = fixture();
    desktop.import(&p, dir.path()).unwrap();
    let old = dir.path().join("old.sacri-library");
    desktop.export_library(&old).unwrap();
    let mut phone = Archive::open(dir.path().join("phone")).unwrap();
    phone.import_portable(&old).unwrap();
    phone
        .assess(
            "evidence-a",
            "uncertain",
            "A note preserved across revisions.",
        )
        .unwrap();
    p.revision = "revision-b".into();
    p.evidence[0].id = "evidence-b".into();
    p.evidence[0].text = "New current material.".into();
    p.claims.clear();
    p.concepts.clear();
    desktop.import(&p, dir.path()).unwrap();
    let next = dir.path().join("next.sacri-library");
    desktop.export_library(&next).unwrap();
    assert!(phone
        .import_portable(&next)
        .unwrap()
        .source_conflicts
        .is_empty());
    assert_eq!(phone.snapshot().unwrap().heads["source-a"], "revision-b");
    phone.import_portable(&old).unwrap();
    assert_eq!(phone.snapshot().unwrap().heads["source-a"], "revision-b");
    assert_eq!(
        phone.search("material", None, None, 6).unwrap()[0]
            .evidence
            .id,
        "evidence-b"
    );
    assert_eq!(
        phone.summary().unwrap().assessments[0].note,
        "A note preserved across revisions."
    );
}

#[test]
fn backup_record_collision_rolls_back_all_knowledge_changes() {
    let dir = tempfile::tempdir().unwrap();
    let mut local = Archive::open(dir.path().join("library")).unwrap();
    local.import(&fixture(), dir.path()).unwrap();
    let mut snapshot = local.snapshot().unwrap();
    let mut new_claim = snapshot.claims[0].clone();
    new_claim.id = "claim-new".into();
    new_claim.statement = "A new claim before the collision.".into();
    snapshot.claims[0].statement = "Conflicting rewrite of preserved claim.".into();
    snapshot.claims.insert(0, new_claim);
    let path = dir.path().join("collision.sacri-library");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    zip.start_file("manifest.json", zip::write::SimpleFileOptions::default())
        .unwrap();
    zip.write_all(&serde_json::to_vec(&snapshot).unwrap())
        .unwrap();
    zip.finish().unwrap();
    assert!(local.import_portable(&path).is_err());
    let summary = local.summary().unwrap();
    assert_eq!(summary.claims.len(), 1);
    assert_eq!(
        summary.claims[0].statement,
        "Collective attention can create an egregore."
    );
}

#[test]
fn corrupt_library_artifact_cannot_add_a_source_or_annotation() {
    let dir = tempfile::tempdir().unwrap();
    let mut original = Archive::open(dir.path().join("original")).unwrap();
    std::fs::write(dir.path().join("captions.json"), b"original").unwrap();
    let source = dir.path().join("source.sacri");
    write_package(
        fixture(),
        &[AssetInput {
            id: "captions".into(),
            name: "captions.json".into(),
            kind: "captions".into(),
            path: dir.path().join("captions.json"),
        }],
        &source,
    )
    .unwrap();
    original.import_file(&source).unwrap();
    original
        .assess("source-a", "collected", "Preserve this.")
        .unwrap();
    let snapshot = original.snapshot().unwrap();
    let asset = &snapshot.assets[0].asset;
    let path = dir.path().join("corrupt.sacri-library");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    let options = zip::write::SimpleFileOptions::default();
    zip.start_file("manifest.json", options).unwrap();
    zip.write_all(&serde_json::to_vec(&snapshot).unwrap())
        .unwrap();
    zip.start_file(format!("assets/{}", asset.sha256), options)
        .unwrap();
    zip.write_all(b"modified").unwrap();
    zip.finish().unwrap();
    let mut destination = Archive::open(dir.path().join("destination")).unwrap();
    assert!(destination.import_portable(&path).is_err());
    assert!(destination.summary().unwrap().sources.is_empty());
    assert!(destination.summary().unwrap().assessments.is_empty());
}

#[test]
fn extractive_drafts_preserve_evidence_and_survive_repeated_imports() {
    let mut p=fixture();p.claims.clear();p.concepts.clear();
    let original=p.evidence[0].clone();p.evidence.clear();
    for i in 0..3 { let mut e=original.clone();e.id=format!("evidence-{i}");e.text="The excerpt discusses symbolic egregores and collective attention without authenticating metaphysical claims.".into();p.evidence.push(e); }
    let raw=serde_json::to_string(&p.evidence).unwrap();navigation_drafts(&mut p);
    assert_eq!(p.claims.len(),3);assert!(!p.concepts.is_empty());
    assert!(p.claims.iter().all(|c| c.kind=="source_excerpt" && c.provenance.method.ends_with("unreviewed")));
    assert_eq!(serde_json::to_string(&p.evidence).unwrap(),raw);p.validate().unwrap();
    let dir=tempfile::tempdir().unwrap();let mut a=Archive::open(dir.path()).unwrap();
    a.import(&p,dir.path()).unwrap();a.import(&p,dir.path()).unwrap();
    assert_eq!(a.summary().unwrap().claims.len(),3);
    a.assess(&p.claims[0].id,"contested","My review").unwrap();navigation_drafts(&mut p);a.import(&p,dir.path()).unwrap();
    assert!(a.summary().unwrap().assessments.iter().any(|v|v.note=="My review"));
}

#[test]
fn annotations_follow_reprocessed_passages_without_overwriting_new_work() {
    let dir = tempfile::tempdir().unwrap();
    let mut archive = Archive::open(dir.path()).unwrap();
    let mut first = fixture();
    first.claims.clear();
    first.concepts.clear();
    first.source.duration_seconds = 700.0;
    let provenance = first.evidence[0].provenance.clone();
    let speech = |id: &str, start: f64, end: f64, text: &str| Evidence {
        id: id.into(), source_id: "source-a".into(), start, end, text: text.into(), kind: EvidenceKind::Speech,
        provenance: provenance.clone(), vector: vec![0.0; EMBEDDING_DIM],
    };
    first.evidence = vec![
        speech("old-dialogue", 430.0, 470.0, "Jeffrey Epstein did not have a bank account named Ball. Yes, he did. This premise is hobgoblins."),
        speech("old-claim", 491.0, 504.0, "So the only time the account is named Ball is when it's actually named Ball."),
        speech("old-unrelated", 600.0, 640.0, "A different subject about temples and baths."),
    ];
    archive.import(&first, dir.path()).unwrap();
    archive.assess("old-dialogue", "contested", "Two voices disagree here.").unwrap();
    archive.correct_reading("old-claim", Some("So the only time the account is named Baal is when it's actually named Baal.")).unwrap();
    archive.add_concept("Account name", "old-claim").unwrap();

    // Reprocessing with voice separation splits the dialogue and labels the claim.
    let mut second = first.clone();
    second.revision = "revision-b".into();
    second.evidence = vec![
        speech("new-denial", 430.0, 439.0, "[Voice B] Jeffrey Epstein did not have a bank account named Ball."),
        speech("new-retort", 439.0, 442.0, "[Voice A] Yes, he did. This premise is hobgoblins."),
        speech("new-claim", 491.0, 504.0, "[Voice A] So the only time the account is named Ball is when it's actually named Ball."),
        speech("new-unrelated", 600.0, 640.0, "[Voice A] A different subject about temples and baths."),
    ];
    archive.assess("old-unrelated", "supported", "Checked.").unwrap();
    archive.import(&second, dir.path()).unwrap();

    let assessments: std::collections::HashMap<String, Assessment> =
        archive.summary().unwrap().assessments.into_iter().map(|a| (a.entity_id.clone(), a)).collect();
    for id in ["new-denial", "new-retort"] {
        assert_eq!(assessments[id].status, "contested");
        assert!(assessments[id].note.starts_with("Two voices disagree here.") && assessments[id].note.contains("Carried forward"));
        assert_eq!(assessments[id].corrected_text, None);
    }
    // The corrected reading keeps the new voice label.
    assert_eq!(assessments["new-claim"].corrected_text.as_deref(), Some("[Voice A] So the only time the account is named Baal is when it's actually named Baal."));
    assert_eq!(archive.reading("new-claim").unwrap(), "[Voice A] So the only time the account is named Baal is when it's actually named Baal.");
    // Concept links follow; old annotations stay on the preserved passages.
    let concept = archive.summary().unwrap().concepts.into_iter().find(|c| c.label == "Account name").unwrap();
    assert!(concept.evidence_ids.contains(&"new-claim".to_string()) && concept.evidence_ids.contains(&"old-claim".to_string()));
    assert_eq!(assessments["old-dialogue"].note, "Two voices disagree here.");
    assert_eq!(assessments["new-unrelated"].status, "supported");
    // A word-for-word correction follows the words even when passage boundaries move.
    let mut fourth = second.clone();
    fourth.revision = "revision-d".into();
    fourth.evidence[2] = speech("moved-claim", 495.0, 510.0, "[Voice A] the account is named Ball is when it's actually named Ball. Now, a search.");
    archive.import(&fourth, dir.path()).unwrap();
    assert_eq!(archive.reading("moved-claim").unwrap(), "[Voice A] the account is named Baal is when it's actually named Baal. Now, a search.");
    let mut second = second.clone();
    second.revision = "revision-e".into();
    archive.import(&second, dir.path()).unwrap();
    // A new passage's own annotation is never replaced by a later carry.
    archive.assess("new-claim", "belief", "My new reading.").unwrap();
    let mut third = second.clone();
    third.revision = "revision-c".into();
    archive.import(&third, dir.path()).unwrap();
    let kept = archive.summary().unwrap().assessments.into_iter().find(|a| a.entity_id == "new-claim").unwrap();
    assert_eq!((kept.status.as_str(), kept.note.as_str()), ("belief", "My new reading."));
    // Search sees the carried reading.
    assert!(archive.search("Baal", None, None, 6).unwrap().iter().any(|h| h.evidence.id == "new-claim"));
}
