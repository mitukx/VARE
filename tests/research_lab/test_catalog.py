from vare.research_lab.spec import TASK_TYPES, load_catalog, repo_root, suite_manifest


def test_catalog_has_three_task_types():
    tasks = load_catalog()
    assert set(tasks) == {
        "grad_accum_scale_v1",
        "loop_invariant_score_v1",
        "descent_sign_debug_v1",
    }
    assert {task.task_type for task in tasks.values()} == set(TASK_TYPES)
    manifest = suite_manifest()
    assert manifest["manifest_sha256"]
    assert len(manifest["files"]) >= 9


def test_task_packs_do_not_embed_hidden_seeds():
    root = repo_root() / "benchmarks" / "research_lab_v1"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*") if path.is_file())
    for seed in ("44011", "44029", "44087"):
        assert seed not in text
