import numpy as np

import dyooptix.geom.object_edit as oe

GREY, RED = (128, 128, 128), (255, 0, 0)


def scene():
    """Grey 200 x 200 photo with a red 40 x 40 square at x 40..79, y 80..119."""
    image = np.full((200, 200, 3), GREY, np.uint8)
    image[80:120, 40:80] = RED
    mask = np.zeros((200, 200), np.uint8)
    mask[80:120, 40:80] = 255
    return image, mask


def plain_fill(image, mask):               # stands in for LaMa: paints the hole with the background colour
    out = image.copy()
    out[mask > 127] = GREY
    return out


def edit(target, monkeypatch):
    monkeypatch.setattr(oe, "fill", plain_fill)
    image, mask = scene()
    return oe.edit_object(image, mask, target)


def red(pixel):                            # fully red, allowing for rounding
    return pixel[0] > 245 and pixel[1] < 10


def grey(pixel):
    return tuple(pixel) == GREY


def test_move_right_takes_the_object_with_it(monkeypatch):
    out, info = edit([50 / 200, 0, 1, 0, 0], monkeypatch)
    assert np.isclose(info["tx"], 50) and info["warnings"] == []
    assert red(out[100, 110])                          # new place: centre moved from x = 59.5 to 109.5
    assert grey(out[100, 50])                          # old place: filled with background


def test_move_past_the_edge_is_clamped_and_warned(monkeypatch):
    out, info = edit([300 / 200, 0, 1, 0, 0], monkeypatch)
    assert np.isclose(info["tx"], 120) and "right edge" in info["warnings"][0]
    assert red(out[100, 190]) and grey(out[100, 150])  # square now spans x 160..199


def test_double_size_grows_about_the_centroid(monkeypatch):
    out, info = edit([0, 0, 1, 0, np.log(2)], monkeypatch)
    assert red(out[100, 30]) and red(out[100, 89])     # about 80 px wide now, same centre x = 59.5
    assert grey(out[100, 15]) and grey(out[100, 104])


def test_no_edit_returns_the_photo_untouched(monkeypatch):
    image, _ = scene()
    out, info = edit([0, 0, 1, 0, 0], monkeypatch)
    assert (out == image).all()