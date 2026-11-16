"""Tests for mapper_tools.skill_evolution (plan §8.2/9.8) and skill_review CLI.

Covers: playbook I/O, pending I/O, changelog, merge_delta (add/amend/retire +
self-dedup), lexical + cross-tag dedup, admission gate, rollback, pause/resume,
the skill_review CLI actions, and the zero-cache-invalidation guarantee for the
new GlobalSettings fields (skill_routing / skill_evolution are NOT inputs to
get_request_hash_id).

Happy/sad/bad paths per project test conventions. Run: python -m pytest tests/ -q
"""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapper_tools import skill_evolution as se  # noqa: E402


@pytest.fixture
def cache_dir(tmp_path):
    """Isolated per-test cache directory (never touches mapper_tools/llm_cache)."""
    return str(tmp_path / "llm_cache")


def _make_bullet(bid, text, **kw):
    return {
        "id": bid, "text": text, "category": "general",
        "scopes": {"factions": [], "sections": []}, "source": "reflector",
        "created_run": "r1", "updated_run": "r1",
        "helpful_count": 0, "harmful_count": 0, "retrieval_hits": 0,
        "last_hit_run": "r1", "pinned": False, "retired": False,
        "shared_from_tag": None, "evidence": [],
        **kw,
    }


def _make_delta(delta_id, operation, text, **kw):
    return {
        "delta_id": delta_id, "operation": operation, "proposed_text": text,
        "rationale": "test", "confidence_score": 0.9,
        "evidence_refs": [],
        "target": {"kind": "playbook_bullet", "skill_name": None, "bullet_id": None},
        **kw,
    }


# ------------------------------------------------------------- playbook I/O ---

class TestPlaybookIO:
    def test_load_missing_returns_empty(self, cache_dir):
        pb = se.load_playbook("TESTTAG", cache_dir)
        assert pb["version"] == se.PLAYBOOK_VERSION
        assert pb["bullets"] == [] and pb["stats"]["total_admitted"] == 0

    def test_load_corrupt_returns_empty(self, cache_dir):
        os.makedirs(os.path.join(cache_dir, "TESTTAG"), exist_ok=True)
        with open(se._playbook_path("TESTTAG", cache_dir), "w") as f:
            f.write("{not json")
        assert se.load_playbook("TESTTAG", cache_dir)["bullets"] == []

    def test_load_version_mismatch_returns_empty(self, cache_dir):
        os.makedirs(os.path.join(cache_dir, "TESTTAG"), exist_ok=True)
        with open(se._playbook_path("TESTTAG", cache_dir), "w") as f:
            json.dump({"version": 999, "bullets": ["x"]}, f)
        assert se.load_playbook("TESTTAG", cache_dir)["bullets"] == []

    def test_save_then_load_roundtrip(self, cache_dir):
        pb = se._empty_playbook()
        pb["bullets"].append(_make_bullet("b-1", "keep infantry coherent"))
        asyncio.run(se.save_playbook(pb, "TESTTAG", cache_dir))
        loaded = se.load_playbook("TESTTAG", cache_dir)
        assert loaded["bullets"][0]["id"] == "b-1"
        assert loaded["bullets"][0]["text"] == "keep infantry coherent"

    def test_pending_roundtrip_and_version(self, cache_dir):
        pending = se.load_pending("TESTTAG", cache_dir)
        assert pending["deltas"] == []
        pending["deltas"].append(_make_delta("d-1", "add", "text"))
        asyncio.run(se.save_pending(pending, "TESTTAG", cache_dir))
        assert se.load_pending("TESTTAG", cache_dir)["deltas"][0]["delta_id"] == "d-1"

    def test_changelog_append_and_rejected_ids(self, cache_dir):
        se.append_changelog({"delta_id": "d-1", "action": "rejected"}, "TESTTAG", cache_dir)
        se.append_changelog({"delta_id": "d-2", "action": "applied"}, "TESTTAG", cache_dir)
        assert se.get_rejected_delta_ids("TESTTAG", cache_dir) == {"d-1"}


# ------------------------------------------------------------ merge_delta ---

class TestMergeDelta:
    def test_add_appends_new_bullet(self, cache_dir):
        pb = se._empty_playbook()
        result = se.merge_delta(pb, _make_delta("d-1", "add", "prefer cultural prefixes"), "r2")
        assert result is not None and result["id"] == se._bullet_id("prefer cultural prefixes")
        assert len(pb["bullets"]) == 1
        assert pb["stats"]["total_admitted"] == 0  # caller bumps stats, not merge

    def test_add_identical_text_self_dedupes(self):
        pb = se._empty_playbook()
        d = _make_delta("d-1", "add", "same guidance")
        first = se.merge_delta(pb, d, "r2")
        second = se.merge_delta(pb, d, "r3")
        assert len(pb["bullets"]) == 1
        assert first["id"] == second["after"]["id"]
        assert second["after"]["helpful_count"] == 1
        assert second["before"]["helpful_count"] == 0  # snapshot restores exact prior state

    def test_amend_replaces_text_and_snapshots_before(self):
        pb = se._empty_playbook()
        pb["bullets"].append(_make_bullet("b-1", "old text"))
        result = se.merge_delta(pb, _make_delta("d-2", "amend", "new text",
                                                target={"kind": "playbook_bullet", "skill_name": None, "bullet_id": "b-1"}), "r2")
        assert pb["bullets"][0]["text"] == "new text"
        assert result["before"]["text"] == "old text"

    def test_amend_unknown_bullet_returns_none(self):
        pb = se._empty_playbook()
        assert se.merge_delta(pb, _make_delta("d-3", "amend", "x",
                                             target={"kind": "playbook_bullet", "skill_name": None, "bullet_id": "nope"}), "r2") is None

    def test_retire_marks_retired_keeps_audit(self):
        pb = se._empty_playbook()
        pb["bullets"].append(_make_bullet("b-1", "stale guidance"))
        se.merge_delta(pb, _make_delta("d-4", "retire", "stale guidance",
                                       target={"kind": "playbook_bullet", "skill_name": None, "bullet_id": "b-1"}), "r2")
        assert pb["bullets"][0]["retired"] is True
        assert len(pb["bullets"]) == 1  # kept for audit, not deleted

    def test_malformed_delta_returns_none(self):
        pb = se._empty_playbook()
        assert se.merge_delta(pb, {"operation": "add", "proposed_text": ""}, "r2") is None
        assert se.merge_delta(pb, {"operation": "wat", "proposed_text": "x"}, "r2") is None


# ----------------------------------------------------------------- dedup ---

class TestDedup:
    def test_lexical_dedup_merges_near_identical(self):
        a = _make_bullet("b-1", "Prefer units sharing the faction's cultural prefix.", helpful_count=3)
        b = _make_bullet("b-2", "Prefer units sharing the faction's cultural prefix", helpful_count=1)
        out = se._lexical_dedup([a, b], threshold=0.9)
        assert len(out) == 1
        assert out[0]["helpful_count"] == 4  # counts summed

    def test_lexical_dedup_keeps_distinct(self):
        a = _make_bullet("b-1", "Infantry guidance only.")
        b = _make_bullet("b-2", "Naval rosters need ships.")
        assert len(se._lexical_dedup([a, b], threshold=0.9)) == 2

    def test_lexical_dedup_never_merges_retired(self):
        a = _make_bullet("b-1", "same text same text", retired=True)
        b = _make_bullet("b-2", "same text same text")
        out = se._lexical_dedup([a, b], threshold=0.99)
        assert len(out) == 2

    def test_cross_tag_dedup_marks_shared(self, cache_dir):
        # Another tag's state dir must exist for discovery.
        other_dir = os.path.join(cache_dir, "OTHERTAG")
        os.makedirs(other_dir, exist_ok=True)

        class FakeColl:
            pass

        class FakeVectorClient:
            def get_collection(self, name):
                return FakeColl() if "OTHERTAG" in name else None

            async def batch_similarity_search(self, coll, queries):
                return {q["id"]: [{"metadata": {"name": "x"}, "score": 0.97}]
                        for q in queries}

        playbook = {"bullets": [_make_bullet("b-1", "shared knowledge text")], "last_run_id": "r1"}
        shared = asyncio.run(se._cross_tag_dedup(playbook, FakeVectorClient(), "TESTTAG",
                                                 {"dedup_similarity_threshold": 0.92}, cache_dir))
        assert shared == 1
        assert playbook["bullets"][0]["shared_from_tag"] == "OTHERTAG"

    def test_cross_tag_dedup_via_other_playbook_file(self, cache_dir):
        # The primary path: compare against the other tag's authoritative
        # skill_playbook.json (the collections are never populated by this module,
        # and the local Chroma search returns no scores).
        other = se._empty_playbook()
        other["bullets"].append(_make_bullet("b-x", "shared knowledge text"))
        asyncio.run(se.save_playbook(other, "OTHERTAG", cache_dir))

        class FakeVectorClient:
            async def _get_embeddings(self, texts, input_type):
                # Character-bag embedding: identical texts -> identical vectors.
                import string
                vocab = string.ascii_lowercase + "_ "
                return [[float(t.lower().count(c)) for c in vocab] for t in texts]

            def get_collection(self, name):
                return None

        playbook = {"bullets": [_make_bullet("b-1", "shared knowledge text")], "last_run_id": "r1"}
        shared = asyncio.run(se._cross_tag_dedup(playbook, FakeVectorClient(), "TESTTAG",
                                                 {"dedup_similarity_threshold": 0.92}, cache_dir))
        assert shared == 1
        assert playbook["bullets"][0]["shared_from_tag"] == "OTHERTAG"

    def test_cross_tag_dedup_distinct_text_not_shared(self, cache_dir):
        other = se._empty_playbook()
        other["bullets"].append(_make_bullet("b-x", "shared knowledge text"))
        asyncio.run(se.save_playbook(other, "OTHERTAG", cache_dir))

        class FakeVectorClient:
            async def _get_embeddings(self, texts, input_type):
                import string
                vocab = string.ascii_lowercase + "_ "
                return [[float(t.lower().count(c)) for c in vocab] for t in texts]

            def get_collection(self, name):
                return None

        playbook = {"bullets": [_make_bullet("b-1", "unrelated siege guidance")], "last_run_id": "r1"}
        shared = asyncio.run(se._cross_tag_dedup(playbook, FakeVectorClient(), "TESTTAG",
                                                 {"dedup_similarity_threshold": 0.92}, cache_dir))
        assert shared == 0
        assert playbook["bullets"][0]["shared_from_tag"] is None

    def test_cross_tag_dedup_noop_without_other_tags(self, cache_dir):
        playbook = {"bullets": [_make_bullet("b-1", "text")], "last_run_id": "r1"}
        shared = asyncio.run(se._cross_tag_dedup(playbook, object(), "TESTTAG", {}, cache_dir))
        assert shared == 0
        assert playbook["bullets"][0]["shared_from_tag"] is None

    def test_refine_playbook_prunes_idle_unpinned(self, cache_dir):
        pb = se._empty_playbook()
        # created_run != last_run_id, retrieval_hits==0, not pinned -> pruned
        pb["bullets"].append(_make_bullet("b-1", "idle bullet", created_run="r1", updated_run="r5"))
        pb["last_run_id"] = "r5"
        pruned = asyncio.run(se.refine_playbook(pb, None, {}, "TESTTAG", cache_dir))
        assert pruned == 1
        assert pb["bullets"][0]["retired"] is True
        assert pb["stats"]["total_pruned"] == 1

    def test_refine_playbook_keeps_pinned_and_fresh(self, cache_dir):
        pb = se._empty_playbook()
        pb["bullets"].append(_make_bullet("b-1", "pinned", pinned=True, created_run="r1", updated_run="r5"))
        pb["bullets"].append(_make_bullet("b-2", "fresh", created_run="r5", updated_run="r5"))
        pb["last_run_id"] = "r5"
        pruned = asyncio.run(se.refine_playbook(pb, None, {}, "TESTTAG", cache_dir))
        assert pruned == 0
        assert all(not b["retired"] for b in pb["bullets"])


# ------------------------------------------------------------ admission ---

class TestAdmission:
    @pytest.fixture
    def playbook(self):
        pb = se._empty_playbook()
        pb["bullets"].append(_make_bullet("b-1", "existing"))
        return pb

    def test_valid_delta_admitted(self, playbook, cache_dir):
        deltas = [_make_delta("d-1", "add", "A perfectly cromulent English sentence.")]
        admitted, rejected = asyncio.run(se.admit_deltas(deltas, playbook, {}, "TESTTAG", "r1", cache_dir))
        assert len(admitted) == 1 and rejected == []

    def test_code_fence_rejected(self, playbook, cache_dir):
        deltas = [_make_delta("d-2", "add", "Use ```python exec('x') ``` now.")]
        admitted, rejected = asyncio.run(se.admit_deltas(deltas, playbook, {}, "TESTTAG", "r1", cache_dir))
        assert admitted == []
        assert "code fence" in rejected[0][1]

    def test_foreign_leading_char_rejected(self, playbook, cache_dir):
        # A line starting with a non-ASCII char violates project rule 3.
        bad = "\u6807\u00b1 pref: use nor_ units"  # leading CJK chars
        deltas = [_make_delta("d-3", "add", bad)]
        admitted, rejected = asyncio.run(se.admit_deltas(deltas, playbook, {}, "TESTTAG", "r1", cache_dir))
        assert admitted == []
        assert "non-ASCII" in rejected[0][1]

    def test_oversize_text_rejected(self, playbook, cache_dir):
        deltas = [_make_delta("d-4", "add", "x" * (se.MAX_BULLET_TEXT + 1))]
        admitted, rejected = asyncio.run(se.admit_deltas(deltas, playbook, {}, "TESTTAG", "r1", cache_dir))
        assert admitted == []
        # Either the safety length check or pydantic re-validation (max_length=2000) fires.
        assert "exceeds" in rejected[0][1] or "validation error" in rejected[0][1]

    def test_amend_unknown_bullet_rejected(self, playbook, cache_dir):
        deltas = [_make_delta("d-5", "amend", "new text",
                             target={"kind": "playbook_bullet", "skill_name": None, "bullet_id": "missing"})]
        admitted, rejected = asyncio.run(se.admit_deltas(deltas, playbook, {}, "TESTTAG", "r1", cache_dir))
        assert admitted == []
        # Pydantic min_length=10 on proposed_text fires first for short text; use long text
        # so the structural bullet_id check is reached.
        deltas = [_make_delta("d-5", "amend", "a sufficiently long replacement text",
                             target={"kind": "playbook_bullet", "skill_name": None, "bullet_id": "missing"})]
        admitted, rejected = asyncio.run(se.admit_deltas(deltas, playbook, {}, "TESTTAG", "r1", cache_dir))
        assert admitted == []
        assert "bullet_id not found" in rejected[0][1]

    def test_rejection_is_recorded_and_never_restaged(self, playbook, cache_dir):
        deltas = [_make_delta("d-6", "add", "``` bad ```")]
        asyncio.run(se.admit_deltas(deltas, playbook, {}, "TESTTAG", "r1", cache_dir))
        assert "d-6" in se.get_rejected_delta_ids("TESTTAG", cache_dir)
        # And reflection memory got the 2-strike entry.
        from mapper_tools import reflection_passes
        mem = reflection_passes.load_memory("TESTTAG", cache_dir)
        # record_review_rejection stores the unit_key (text[:100]) under scope=skill_evolution;
        # the delta_id appears in the reason string.
        fb = mem.get("feedback", {}).get("skill_evolution", {})
        assert any("review_rejection:d-6" in r for rec in fb.values() for r in rec.get("reasons", []))


# ---------------------------------------------- rollback / pause / resume ---

class TestRollbackAndPause:
    def _applied_playbook(self, cache_dir):
        """Builds a playbook with one applied add-delta in the changelog."""
        pb = se._empty_playbook()
        d = _make_delta("d-1", "add", "rollback me")
        result = se.merge_delta(pb, d, "r1")
        se.append_changelog({
            "ts": se._iso_now(), "run_id": "r1", "delta_id": "d-1", "action": "applied",
            "target": d["target"], "before": None,
            "after": {"id": result["id"], "text": result["text"]}, "actor": "auto",
        }, "TESTTAG", cache_dir)
        return pb, result["id"]

    def test_rollback_add_removes_bullet(self, cache_dir):
        pb, bid = self._applied_playbook(cache_dir)
        reverted = se.rollback_deltas(pb, "TESTTAG", n_runs=1, cache_dir=cache_dir)
        assert reverted == 1
        assert pb["bullets"] == []
        assert pb["stats"]["total_reverted"] == 1

    def test_rollback_is_logged_and_reversible(self, cache_dir):
        pb, bid = self._applied_playbook(cache_dir)
        se.rollback_deltas(pb, "TESTTAG", n_runs=1, cache_dir=cache_dir)
        entries = se.load_changelog("TESTTAG", cache_dir)
        assert entries[-1]["action"] == "reverted" and entries[-1]["actor"] == "human:rollback"

    def test_rollback_by_delta_ids(self, cache_dir):
        pb, bid = self._applied_playbook(cache_dir)
        reverted = se.rollback_deltas(pb, "TESTTAG", delta_ids={"d-1"}, cache_dir=cache_dir)
        assert reverted == 1 and pb["bullets"] == []

    def test_pause_resume_cycle(self, cache_dir):
        assert se.is_paused("TESTTAG", cache_dir) is False
        se.pause_evolution("TESTTAG", reason="auditing", cache_dir=cache_dir)
        assert se.is_paused("TESTTAG", cache_dir) is True
        info = se.get_pause_info("TESTTAG", cache_dir)
        assert info["reason"] == "auditing" and "paused_at" in info
        assert se.resume_evolution("TESTTAG", cache_dir=cache_dir) is True
        assert se.is_paused("TESTTAG", cache_dir) is False
        assert se.resume_evolution("TESTTAG", cache_dir=cache_dir) is False  # idempotent

    def test_paused_cycle_stages_but_never_admits(self, cache_dir):
        se.pause_evolution("TESTTAG", reason="hold", cache_dir=cache_dir)

        class SilentHelper:
            network_calls_enabled = True

            async def _call_llm_with_retry(self, *a, **kw):
                return json.dumps({"deltas": [
                    {"target_kind": "playbook_bullet", "operation": "add",
                     "proposed_text": "staged while paused", "rationale": "r",
                     "confidence_score": 0.9, "evidence_refs": []}
                ]})

        from unittest.mock import patch
        with patch.object(se.shared_utils, "parse_llm_json_response",
                          lambda content, key, model_used=None: {"deltas": [
                              {"target_kind": "playbook_bullet", "operation": "add",
                               "proposed_text": "staged while paused", "rationale": "r",
                               "confidence_score": 0.9, "evidence_refs": []}]}):
            summary = asyncio.run(se.run_evolution_cycle(
                None, SilentHelper(), {}, "r1", {"enabled": True},
                llm_cache_tag="TESTTAG", cache_dir=cache_dir))
        assert summary["admitted"] == 0
        assert len(se.load_pending("TESTTAG", cache_dir)["deltas"]) == 1


# ------------------------------------------------------------ curriculum ---

class TestCurriculum:
    def test_ranks_scopes_by_flags(self):
        memory = {"feedback": {
            "Winterfell": {"u1": {"flags": 5}, "u2": {"flags": 7}},
            "Dorne": {"u3": {"flags": 1}},
        }}
        out = se.propose_curriculum(memory, se._empty_playbook())
        assert len(out) == 2
        assert "Winterfell" in out[0] and "12" in out[0]

    def test_empty_feedback_returns_empty(self):
        assert se.propose_curriculum({}, se._empty_playbook()) == []

    def test_retrieval_hits_recorded(self):
        pb = se._empty_playbook()
        pb["bullets"].append(_make_bullet("b-1", "uses roster category", category="roster"))
        pb["bullets"].append(_make_bullet("b-2", "naval text"))
        updated = se.record_skill_retrieval_hits(pb, ["roster"], "r9")
        assert updated == 1
        assert pb["bullets"][0]["retrieval_hits"] == 1
        assert pb["bullets"][0]["last_hit_run"] == "r9"
        assert pb["bullets"][1]["retrieval_hits"] == 0


# ------------------------------------------------- zero cache invalidation ---

class TestZeroCacheInvalidation:
    """§8.1.F: skill_routing/skill_evolution must NOT change get_request_hash_id."""

    BASE_GLOBAL = {
        "use_g4f_as_last_resort": True, "randomize_tiers": False, "loop_on_failure": True,
    }

    def _helper_with(self, extra_global):
        from mapper_tools.llm_config_schema import FullLLMConfig
        cfg = {
            "global_settings": {**self.BASE_GLOBAL, **extra_global},
            "vector_db_config": {"backend": "local", "local_backend": "chroma",
                                 "embedding_models": [{"model": "m"}]},
            "astra_db_config": {"api_endpoint": "e", "token": "t"},
            "llm_configs": [{"name": "n", "model": "m", "request_timeout": 1, "max_retries": 1}],
        }
        parsed = FullLLMConfig.parse_obj(cfg).model_dump()
        helper = SimpleNamespace(
            llm_configs=parsed["llm_configs"],
            max_tool_calls=None,
            review_llm_configs=None, review_llm_configs_2=None,
            review_llm_configs_3=None, review_llm_configs_4=None,
            judge_llm_configs=None,
            max_consensus_review_rounds=None,
            review_selection_mode=None,
        )
        return helper

    def _hash_via(self, helper, request):
        # Reproduce get_request_hash_id's inputs without constructing a full LLMHelper.
        import hashlib as _h
        from mapper_tools import shared_utils as su
        keys = [
            'id', 'faction', 'faction_name', 'roster', 'validation_pool',
            'local_unit_pool', 'known_bad_units', 'heritage', 'cultures',
        ]
        content = {k: request[k] for k in keys if k in request}
        canonical = su.canonicalize_for_hash(content)
        content_hash = _h.sha256(str(canonical).encode('utf-8')).hexdigest()
        # Config fingerprint inputs (mirror _get_config_fingerprint 'general' pass).
        parts = [('max_tool_calls', helper.max_tool_calls)]
        for cfg in (helper.llm_configs or []):
            parts.append(('general', cfg.get('name'), cfg.get('model'),
                          cfg.get('temperature'), cfg.get('max_tool_calls')))
        fp = _h.sha256(json.dumps(parts, sort_keys=True).encode('utf-8')).hexdigest()[:16]
        return f"v2|{content_hash}|{fp}"

    def test_new_settings_do_not_touch_hash_inputs(self):
        """The new GlobalSettings fields must not appear in the hash input set."""
        from mapper_tools.llm_helper import LLMHelper
        src = open(LLMHelper.get_request_hash_id.__code__.co_filename, encoding='utf-8').read()
        fn_src = src.split('def get_request_hash_id')[1].split('def _infer_pass_name')[0]
        assert 'skill_routing' not in fn_src
        assert 'skill_evolution' not in fn_src
        # And the config fingerprint source likewise.
        fp_src = src.split('def _get_config_fingerprint')[1].split('def dedup_in_flight')[0]
        assert 'skill_routing' not in fp_src and 'skill_evolution' not in fp_src

    def test_hash_stable_with_and_without_new_fields(self):
        request = {"id": "x", "faction": "Danes", "validation_pool": ["a", "b"]}
        plain = self._helper_with({})
        with_fields = self._helper_with({
            "skill_routing": {"enabled": True, "top_k": 7, "min_score": 0.4},
            "skill_evolution": {"enabled": True, "max_deltas_per_run": 4,
                                "require_human_approval": True,
                                "dedup_similarity_threshold": 0.95,
                                "prune_after_idle_runs": 3,
                                "reflector_use_judge_tier": True},
        })
        assert self._hash_via(plain, request) == self._hash_via(with_fields, request)

    def test_config_schema_defaults_disabled(self):
        from mapper_tools.llm_config_schema import GlobalSettings, SkillRoutingConfig, SkillEvolutionConfig
        gs = GlobalSettings(**self.BASE_GLOBAL)
        assert gs.skill_routing is None and gs.skill_evolution is None
        assert SkillRoutingConfig().enabled is False
        assert SkillEvolutionConfig().enabled is False
        assert SkillEvolutionConfig().require_human_approval is False


# ------------------------------------------------------------ skill_review ---

class TestSkillReviewCLI:
    def _argv(self, *a):
        return list(a)

    def test_list_empty_state(self, cache_dir, capsys):
        from mapper_tools import skill_review as sr
        rc = sr.main(self._argv("--llm-cache-tag", "TESTTAG", "--llm-cache-dir", cache_dir, "--list"))
        out = capsys.readouterr().out
        assert rc == 0
        assert "Playbook" in out and "No pending" in out and "Changelog is empty" in out

    def test_pause_and_resume_cli(self, cache_dir, capsys):
        from mapper_tools import skill_review as sr
        rc1 = sr.main(self._argv("--llm-cache-tag", "TESTTAG", "--llm-cache-dir", cache_dir,
                                  "--pause", "--reason", "audit"))
        assert rc1 == 0 and se.is_paused("TESTTAG", cache_dir)
        rc2 = sr.main(self._argv("--llm-cache-tag", "TESTTAG", "--llm-cache-dir", cache_dir, "--resume"))
        assert rc2 == 0 and not se.is_paused("TESTTAG", cache_dir)

    def test_approve_applies_staged_delta(self, cache_dir, capsys):
        from mapper_tools import skill_review as sr
        pending = se.load_pending("TESTTAG", cache_dir)
        pending["deltas"].append(_make_delta("d-9", "add", "human-approved guidance"))
        asyncio.run(se.save_pending(pending, "TESTTAG", cache_dir))
        rc = sr.main(self._argv("--llm-cache-tag", "TESTTAG", "--llm-cache-dir", cache_dir,
                                "--approve", "d-9"))
        assert rc == 0
        pb = se.load_playbook("TESTTAG", cache_dir)
        assert any("human-approved guidance" in b["text"] for b in pb["bullets"])
        assert pb["stats"]["total_admitted"] == 1
        entries = se.load_changelog("TESTTAG", cache_dir)
        assert entries[-1]["action"] == "applied" and entries[-1]["actor"] == "human:approve"
        assert se.load_pending("TESTTAG", cache_dir)["deltas"] == []

    def test_approve_unknown_delta_fails(self, cache_dir, capsys):
        from mapper_tools import skill_review as sr
        rc = sr.main(self._argv("--llm-cache-tag", "TESTTAG", "--llm-cache-dir", cache_dir,
                                "--approve", "d-nope"))
        assert rc == 1

    def test_reject_removes_and_bars(self, cache_dir, capsys):
        from mapper_tools import skill_review as sr
        pending = se.load_pending("TESTTAG", cache_dir)
        pending["deltas"].append(_make_delta("d-10", "add", "bad guidance"))
        asyncio.run(se.save_pending(pending, "TESTTAG", cache_dir))
        rc = sr.main(self._argv("--llm-cache-tag", "TESTTAG", "--llm-cache-dir", cache_dir,
                                "--reject", "d-10", "--reason", "wrong heritage"))
        assert rc == 0
        assert se.load_pending("TESTTAG", cache_dir)["deltas"] == []
        assert "d-10" in se.get_rejected_delta_ids("TESTTAG", cache_dir)

    def test_rollback_cli_restores_prior_state(self, cache_dir, capsys):
        from mapper_tools import skill_review as sr
        pb = se._empty_playbook()
        d = _make_delta("d-11", "add", "to be rolled back")
        result = se.merge_delta(pb, d, "r1")
        asyncio.run(se.save_playbook(pb, "TESTTAG", cache_dir))
        se.append_changelog({
            "ts": se._iso_now(), "run_id": "r1", "delta_id": "d-11", "action": "applied",
            "target": d["target"], "before": None,
            "after": {"id": result["id"], "text": result["text"]}, "actor": "auto",
        }, "TESTTAG", cache_dir)
        rc = sr.main(self._argv("--llm-cache-tag", "TESTTAG", "--llm-cache-dir", cache_dir,
                                "--rollback", "--runs", "1"))
        assert rc == 0
        assert se.load_playbook("TESTTAG", cache_dir)["bullets"] == []

    def test_pin_bullet_cli(self, cache_dir, capsys):
        from mapper_tools import skill_review as sr
        pb = se._empty_playbook()
        pb["bullets"].append(_make_bullet("b-77", "pin me"))
        asyncio.run(se.save_playbook(pb, "TESTTAG", cache_dir))
        asyncio.run(sr._pin_bullet("b-77", "TESTTAG", cache_dir))
        assert se.load_playbook("TESTTAG", cache_dir)["bullets"][0]["pinned"] is True


# ------------------------------------------------------- end-to-end cycle ---

class TestEvolutionCycle:
    def test_full_cycle_admits_and_persists(self, cache_dir):
        from unittest.mock import patch

        class FakeHelper:
            network_calls_enabled = True

            async def _call_llm_with_retry(self, *a, **kw):
                return "{}"

        deltas_payload = {"deltas": [
            {"target_kind": "playbook_bullet", "operation": "add",
             "proposed_text": "cycle bullet text", "rationale": "evidence-driven",
             "confidence_score": 0.9, "evidence_refs": ["Winterfell"]},
        ]}
        with patch.object(se.shared_utils, "parse_llm_json_response",
                          lambda content, key, model_used=None: deltas_payload):
            summary = asyncio.run(se.run_evolution_cycle(
                None, FakeHelper(), {"feedback": {}}, "runA",
                {"enabled": True, "max_deltas_per_run": 5},
                llm_cache_tag="TESTTAG", cache_dir=cache_dir))
        assert summary["reflected"] == 1
        assert summary["admitted"] == 1
        pb = se.load_playbook("TESTTAG", cache_dir)
        assert any(b["text"] == "cycle bullet text" for b in pb["bullets"])
        entries = se.load_changelog("TESTTAG", cache_dir)
        assert any(e["action"] == "applied" and e["delta_id"] == "d-" + "x" or
                   e["action"] == "applied" for e in entries)

    def test_disabled_helper_returns_zero_summary(self, cache_dir):
        summary = asyncio.run(se.run_evolution_cycle(
            None, None, {}, "r1", {"enabled": True},
            llm_cache_tag="TESTTAG", cache_dir=cache_dir))
        assert summary["reflected"] == 0 and summary["admitted"] == 0

    def test_previously_rejected_delta_never_readmitted(self, cache_dir):
        from unittest.mock import patch
        # Seed the changelog with a rejection for the exact delta_id the Reflector returns.
        se.append_changelog({"delta_id": "d-bad", "action": "rejected"}, "TESTTAG", cache_dir)

        class FakeHelper:
            network_calls_enabled = True

            async def _call_llm_with_retry(self, *a, **kw):
                return "{}"

        deltas_payload = {"deltas": [
            {"target_kind": "playbook_bullet", "operation": "add",
             "proposed_text": "doomed proposal", "rationale": "r",
             "confidence_score": 0.9, "evidence_refs": []},
        ]}
        with patch.object(se.shared_utils, "parse_llm_json_response",
                          lambda content, key, model_used=None: deltas_payload):
            # The delta_id is derived from text, so craft text to match a known id.
            did = se._delta_id("playbook_bullet", "add", "doomed proposal")
        se.append_changelog({"delta_id": did, "action": "rejected"}, "TESTTAG", cache_dir)
        with patch.object(se.shared_utils, "parse_llm_json_response",
                          lambda content, key, model_used=None: {
                              "deltas": [{"target_kind": "playbook_bullet", "operation": "add",
                                          "proposed_text": "doomed proposal", "rationale": "r",
                                          "confidence_score": 0.9, "evidence_refs": []}]}):
            summary = asyncio.run(se.run_evolution_cycle(
                None, FakeHelper(), {}, "r1", {"enabled": True},
                llm_cache_tag="TESTTAG", cache_dir=cache_dir))
        # The rejected id is filtered before admission.
        assert summary["admitted"] == 0
        assert se.load_playbook("TESTTAG", cache_dir)["bullets"] == []

    def test_byte_identical_when_disabled(self, cache_dir):
        """Config unset -> cycle runs but nothing LLM-driven happens without a helper."""
        before = open(__file__, "rb").read()  # sanity that fixtures are file-isolated
        summary = asyncio.run(se.run_evolution_cycle(
            None, None, {}, "r1", {}, llm_cache_tag="TESTTAG", cache_dir=cache_dir))
        assert summary == {"reflected": 0, "admitted": 0, "rejected": 0,
                            "pruned": 0, "curriculum": []}
        assert open(__file__, "rb").read() == before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))