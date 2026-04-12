from ai.roadmap.assistant_gap_analysis import (
    build_current_capability_snapshot,
    summarize_gap_report,
)


def test_gap_snapshot_has_expected_areas():
    areas = build_current_capability_snapshot()
    names = {area.name for area in areas}
    assert "Natural language routing" in names
    assert "Action reliability" in names
    assert len(areas) >= 6


def test_gap_report_prioritizes_largest_gap_first():
    report = summarize_gap_report()
    priorities = report["priority_order"]
    assert priorities
    assert priorities[0]["gap"] >= priorities[-1]["gap"]
    assert report["overall_target"] > report["overall_current"]
