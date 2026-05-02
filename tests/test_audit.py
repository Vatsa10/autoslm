from pathlib import Path
import tempfile
from autoslm.audit import AuditLog


def test_audit_creates_and_appends():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "log.md"
        log = AuditLog(p, run_id="r1", mode="prod")
        log.section("First", "body one")
        log.section("Second", "body two")
        text = p.read_text(encoding="utf-8")
        assert "run_id:" in text and "r1" in text
        assert "## First" in text and "body one" in text
        assert "## Second" in text and "body two" in text
        # subsequent open does not overwrite
        log2 = AuditLog(p)
        log2.section("Third", "body three")
        text2 = p.read_text(encoding="utf-8")
        assert "## First" in text2 and "## Third" in text2
