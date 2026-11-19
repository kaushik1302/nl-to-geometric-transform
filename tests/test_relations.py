from dyooptix.models.relations import name, parse


def test_touch_with_direction():
    r = parse("move the circle right until it touches the square")
    assert (r["kind"], r["mover"], r["target"], r["has_direction"]) == ("touch", "circle", "square", True)
    assert r["rest"] == "move the circle right"


def test_touch_without_direction_and_a_synonym():
    r = parse("push the circle until it reaches the square")
    assert (r["kind"], r["target"], r["has_direction"]) == ("touch", "square", False)


def test_beside_and_its_synonyms_keep_who_moves():
    assert (parse("move the bottle beside the mug")["mover"], parse("move the bottle beside the mug")["target"]) == ("bottle", "mug")
    r = parse("put the mug next to the bottle")
    assert (r["kind"], r["mover"], r["target"], r["side"]) == ("beside", "mug", "bottle", None)


def test_named_side_and_two_word_names():
    r = parse("place the flower pot to the right of the watering can")
    assert (r["mover"], r["target"], r["side"]) == ("flower pot", "watering can", "right")


def test_plain_direction_is_not_a_relation():
    assert parse("move the bottle to the left") is None
    assert parse("move the bottle left by 50 px") is None


def test_object_names():
    assert name("drag the kettle 40 px upward") == "kettle"
    assert name("give the toaster a little push left") == "toaster"
    assert name("rotate the image 90 degrees") is None
    assert name("move it left") is None


def test_edge_words_and_sides():
    r = parse("move the mug until it touches the left end")
    assert (r["kind"], r["mover"], r["side"]) == ("edge", "mug", "left")
    assert parse("slide the mug right until it reaches the right end of the picture")["side"] == "right"
    r = parse("push it up until it hits the upper border")
    assert (r["kind"], r["mover"], r["side"], r["has_direction"]) == ("edge", None, "top", True)
    r = parse("move the mug until it touches the edge")
    assert (r["kind"], r["side"], r["has_direction"]) == ("edge", None, False)