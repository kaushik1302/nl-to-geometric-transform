from dyooptix.models.eval_fresh import factor


def test_percent_to_and_by():
    assert factor("75%", "smaller", "Scale the image to 75% of its size.") == "0.75"
    assert factor("25%", "smaller", "Scale the image by 25%.") == "0.75"
    assert factor("150%", "bigger", "Make the whole picture by 150%.") == "2.5"
    assert factor("130%", "bigger", "Enlarge the cat by 130%.") == "2.3"
    assert factor("1.25", "bigger", "Make it 1.25 times its size.") == "1.25"