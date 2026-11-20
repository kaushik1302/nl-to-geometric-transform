import numpy as np

from dyooptix.geom import grounding

IMAGE = np.zeros((100, 100, 3), np.uint8)
A, B = [10, 10, 50, 50], [60, 60, 90, 90]          # 16% and 9% of the photo


def fake(monkeypatch, found):
    monkeypatch.setattr(grounding, "_detect", lambda image, text: ([s for s, b in found], [b for s, b in found]))
    monkeypatch.setattr(grounding, "_segment", lambda image, box: np.full(image.shape[:2], 255, np.uint8))
    grounding._found.clear()


def test_one_clear_main_object(monkeypatch):
    fake(monkeypatch, [(0.8, A), (0.4, [60, 60, 70, 70]), (0.06, [0, 0, 100, 100])])
    r = grounding.find_main(IMAGE)
    assert r["status"] == "main" and r["box"] == A and len(r["candidates"]) == 1 and r["message"] == ""


def test_scores_about_equal_is_a_tie_and_the_biggest_is_edited(monkeypatch):
    fake(monkeypatch, [(0.80, B), (0.75, A)])
    r = grounding.find_main(IMAGE)
    assert r["status"] == "tie" and r["box"] == A and len(r["candidates"]) == 2
    assert "type an object name" in r["message"]


def test_biggest_wins_over_a_higher_score(monkeypatch):
    fake(monkeypatch, [(0.80, B), (0.60, A), (0.1, [0, 0, 5, 5])])        # 0.20 apart, but A is a similar-sized object
    r = grounding.find_main(IMAGE)
    assert r["status"] == "tie" and r["box"] == A and r["score"] == 0.60


def test_low_score_object_of_similar_size_still_counts(monkeypatch):
    fake(monkeypatch, [(0.80, A), (0.20, B)])
    assert grounding.find_main(IMAGE)["status"] == "tie"


def test_a_box_spanning_the_photo_is_not_a_similar_object(monkeypatch):
    fake(monkeypatch, [(0.80, [0, 20, 60, 80]), (0.20, [0, 50, 95, 100])])
    assert grounding.find_main(IMAGE)["status"] == "main"


def test_no_object_means_whole_image(monkeypatch):
    fake(monkeypatch, [(0.2, A), (0.9, [0, 0, 100, 100])])
    r = grounding.find_main(IMAGE)
    assert r["status"] == "none" and r["mask"] is None and "whole image" in r["message"]


def test_same_object_found_twice_is_not_a_tie(monkeypatch):
    fake(monkeypatch, [(0.80, A), (0.75, [12, 12, 50, 50])])
    assert grounding.find_main(IMAGE)["status"] == "main"