from experiments.atom_decomposition.hint import select, with_hint


def c(key, r):
    return {"key": key, "rate_big": r}


def test_with_hint():
    assert with_hint("Calculate 1 + 2.", "Note: x.") == "Calculate 1 + 2.\n\nNote: x."
    assert with_hint("Calculate 1 + 2.", None) == "Calculate 1 + 2."


def test_in_band():
    s = select([c("c1", 0.004), c("c2", 0.03), c("c3", 0.045), c("c4", 0.12), c("c5", 0.16), c("c6", 0.5)])
    assert (s["weak"]["key"], s["strong"]["key"], s["band_miss"], s["abort"]) == ("c2", "c5", [], None)


def test_nearest_band_when_missed():
    s = select([c("c1", 0.0), c("c2", 0.01), c("c3", 0.06), c("c4", 0.3), c("c5", 0.6)])
    assert (s["weak"]["key"], s["strong"]["key"]) == ("c3", "c4")
    assert s["band_miss"] == ["weak", "strong"]


def test_all_low_still_orders():
    s = select([c("c1", 0.0), c("c2", 0.012), c("c3", 0.018)])
    assert (s["weak"]["key"], s["strong"]["key"]) == ("c2", "c3")


def test_abort():
    assert select([c("c1", 0.0), c("c2", 0.009)])["abort"]
    assert select([c("c1", 0.15), c("c2", 0.15)])["abort"]
