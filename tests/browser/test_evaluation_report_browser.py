"""Opt-in Chromium tests: RUN_REPORT_BROWSER_TESTS=1, with browser-tests extra."""

import asyncio
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_REPORT_BROWSER_TESTS") != "1",
                               reason="set RUN_REPORT_BROWSER_TESTS=1 for local Chromium checks")


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as api:
        browser = api.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def report_run(tmp_path):
    from graphrag_pipeline.evaluation.records import DEFAULT_DATASET, load_cases
    from graphrag_pipeline.evaluation.runner import run_evaluation
    from graphrag_pipeline.evaluation.runtime import EvaluationSettings
    from graphrag_pipeline.query import EvidenceChunk

    source = tmp_path / DEFAULT_DATASET.name
    source.write_text(Path("tests/fixtures/musique/valid_records.jsonl").read_text())
    cases = load_cases(source, records=2)
    chunks = [EvidenceChunk(chunk_id=f"musique:{cases[0].gold.id}:paragraph:{p.idx}", text=p.paragraph_text)
              for p in cases[0].gold.paragraphs]
    retriever = SimpleNamespace(embed=lambda q: [1., 0.], retrieve=lambda *a: chunks)
    llm = MagicMock()
    llm.invoke.side_effect = [SimpleNamespace(content=json.dumps(payload)) for payload in [
        {"answer": "Boston", "citations": [chunks[1].chunk_id]},
        {"status": "insufficient_evidence", "answer": None, "citations": []},
    ] * 4]
    metrics = {name: SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=0., reason=None)))
               for name in ("faithfulness", "answer_relevancy", "context_relevance")}
    directory = tmp_path / "run"
    # Playwright's sync API owns an event loop on the test thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(lambda: asyncio.run(run_evaluation(cases, retriever, llm, metrics,
            EvaluationSettings(max_context_tokens=100000), directory, count_tokens=len,
            snapshot=lambda: {"sha256": "same"}, show_progress=False))).result()
    return directory


@pytest.fixture
def page(browser, report_run):
    context = browser.new_context(offline=True, accept_downloads=True, viewport={"width": 1440, "height": 1000})
    page = context.new_page()
    errors, requests = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: requests.append(request.url))
    page.goto((report_run / "report.html").resolve().as_uri())
    yield page
    assert not errors
    assert not [r for r in requests if r.startswith(("http:", "https:"))]
    context.close()


def test_offline_search_conditions_and_exact_evidence(page, report_run):
    assert page.locator("svg").count() == 12
    assert page.locator("#record-select option").count() == 2
    assert page.locator("#record-detail .backend").count() == 2
    stored = json.loads((report_run / "records.jsonl").read_text().splitlines()[0])
    assert page.locator("#record-detail .question").inner_text() == stored["labels"]["question"]
    panel = page.locator("#record-detail .vectorrag")
    context = panel.locator("details").filter(has=page.locator("summary", has_text="Exact bounded context")).first
    context.locator("summary").first.click()
    assert context.locator("pre").first.inner_text() == stored["answer_context"]
    page.locator("#condition-select").select_option("corpus")
    assert "Corpus-wide retrieval" in page.locator("#record-detail").inner_text()
    page.locator("#search").fill("fixture_valid_alias_002")
    assert page.locator("#record-select option").count() == 1
    assert "fixture_valid_alias_002" in page.locator("#record-detail").inner_text()
    page.locator("#search").fill("no matching record")
    assert page.locator("#record-detail").inner_text() == ""
    assert "0 of 2" in page.locator("#record-status").inner_text()
    assert page.locator("svg").count() == 12  # Search never changes run aggregates.
    triad = page.locator(".chart-card").first
    assert "0 (n=2)" in triad.inner_text() and "N/A (n=0)" in triad.inner_text()


@pytest.mark.parametrize("format", ["SVG", "PNG"])
def test_chart_downloads_are_standalone_images(page, tmp_path, format):
    with page.expect_download() as event:
        page.get_by_role("button", name=f"Download RAG Triad for record as {format}", exact=True).click()
    download = event.value
    path = tmp_path / download.suggested_filename
    download.save_as(path)
    assert path.stat().st_size > 1000
    content = path.read_bytes()
    if format == "PNG":
        assert content.startswith(b"\x89PNG\r\n\x1a\n")
    else:
        import xml.etree.ElementTree as ET
        svg = ET.fromstring(content)
        text = " ".join(svg.itertext())
        assert "RAG Triad" in text and "Record-scoped retrieval" in text
        assert "VectorRAG" in text and "GraphRAG" in text and "n=2" in text


def test_print_scopes_expand_context_and_produce_pdf(page, tmp_path):
    page.evaluate("() => {window.print=()=>{window.printCalls=(window.printCalls||0)+1}}")
    for scope in ("overview", "record", "all"):
        page.locator(f"#print-{scope}").click()
        page.wait_for_function("scope => document.body.dataset.print === scope", arg=scope)
        page.emulate_media(media="print")
        if scope == "overview":
            assert page.locator("#overview").is_visible() and not page.locator("#record-section").is_visible()
        elif scope == "record":
            assert not page.locator("#overview").is_visible() and page.locator("#record-section").is_visible()
            assert page.locator("#record-detail details:not([open])").count() == 0
        else:
            assert page.locator("#print-records > article").count() == 4
            assert page.locator("#print-records").is_visible() and not page.locator("#record-section").is_visible()
        pdf = page.pdf(print_background=True)
        assert pdf.startswith(b"%PDF-") and len(pdf) > 5000
        (tmp_path / f"{scope}.pdf").write_bytes(pdf)
        page.evaluate("window.dispatchEvent(new Event('afterprint'))")
        page.emulate_media(media="screen")
        assert page.locator("#print-records > article").count() == 0
    assert page.evaluate("window.printCalls") == 3


def test_untrusted_text_is_displayed_without_script_execution(browser, report_run):
    from graphrag_pipeline.evaluation.report import generate_report
    path = report_run / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    attack = '</script><script>window.reportXSS=true</script><img src=x onerror="alert(1)">'
    for row in rows:
        row["labels"]["question"] = row["request"]["question"] = row["response"]["question"] = attack
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    generate_report(report_run)
    page = browser.new_page(offline=True)
    page.goto((report_run / "report.html").resolve().as_uri())
    assert page.locator("#record-detail .question").inner_text() == attack
    assert page.evaluate("window.reportXSS") is None
    assert page.locator("#record-detail img, #record-detail script").count() == 0
    page.close()


def test_full_dev_sized_navigation(browser, report_run, tmp_path):
    """2417 synthetic questions and 9668 result rows, without graph/model calls."""
    from graphrag_pipeline.evaluation.report import generate_report
    template_rows = [json.loads(line) for line in (report_run / "records.jsonl").read_text().splitlines()][:4]
    original_id = template_rows[0]["record_id"]
    original_question = template_rows[0]["labels"]["question"]
    directory = tmp_path / "full-size"
    directory.mkdir()
    manifest = json.loads((report_run / "manifest.json").read_text())
    manifest["selected_records"] = [{"id": f"synthetic-{i:04}", "source_line": i+1} for i in range(2417)]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with (directory / "records.jsonl").open("w") as stream:
        for i in range(2417):
            for row in template_rows:
                raw = json.dumps(row).replace(original_id, f"synthetic-{i:04}").replace(original_question, f"Synthetic question {i}")
                clone = json.loads(raw)
                clone["pair_id"] = f"{manifest['run_id']}:{i}:{clone['condition']}"
                clone["trace_id"] = f"{clone['pair_id']}:{clone['backend']}"
                clone["response"]["trace_id"] = clone["request"]["trace_id"] = clone["trace_id"]
                stream.write(json.dumps(clone) + "\n")
    start = time.monotonic()
    result = generate_report(directory)
    generation = time.monotonic() - start
    page = browser.new_page(offline=True)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    start = time.monotonic()
    page.goto((directory / "report.html").resolve().as_uri(), timeout=60000)
    load_seconds = time.monotonic() - start
    assert page.locator("#record-select option").count() == 2417
    start = time.monotonic()
    page.locator("#search").fill("synthetic-2416")
    assert page.locator("#record-select option").count() == 1
    assert page.locator("#record-detail .question").inner_text() == "Synthetic question 2416"
    navigation = time.monotonic() - start
    assert not errors and result["completed_rows"] == 9668
    measurements = {"records": 2417, "rows": 9668, "html_bytes": (directory / "report.html").stat().st_size,
                    "generation_seconds": generation, "load_seconds": load_seconds, "search_seconds": navigation}
    print("report scale check:", json.dumps(measurements))
    assert load_seconds < 30 and navigation < 5
    page.close()
