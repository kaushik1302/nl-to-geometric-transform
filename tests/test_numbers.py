import numpy as np
from dyooptix.models.numbers import fix_numbers

T, R, S = (True, False, False), (False, True, False), (False, False, True)


def pred(tx=0, ty=0, deg=0, logs=0):
    a = np.radians(deg)
    return [tx, ty, np.cos(a), np.sin(a), logs]


def angle(o):
    return np.degrees(np.arctan2(o[3], o[2]))


def test_rotate_clockwise_exact():
    assert np.isclose(angle(fix_numbers("rotate it 20 deg clockwise", pred(deg=-7.5), R)), -20) # type: ignore


def test_rotate_wraps_any_angle():
    assert np.isclose(angle(fix_numbers("rotate it 400 degrees counter-clockwise", pred(deg=10), R)), 40)
    assert np.isclose(angle(fix_numbers("rotate it 400 degrees clockwise", pred(deg=-10), R)), -40)


def test_rotate_without_direction_is_counter_clockwise():
    assert np.isclose(angle(fix_numbers("rotate it 30 degrees", pred(deg=-5), R)), 30)
    assert np.isclose(angle(fix_numbers("rotate it 30", pred(deg=-5), R)), 30)


def test_translate_px_and_percent_keep_direction():
    o = fix_numbers("move it left by 100 px", pred(tx=-0.3, ty=0.01), T) # type: ignore
    assert np.allclose(o[:2], [-100 / 512, 0])
    o = fix_numbers("move it up 25%", pred(tx=0.0, ty=-0.1), T) # type: ignore
    assert np.allclose(o[:2], [0, -0.25])

def test_diagonal_translation_snaps_and_uses_total_length():
    o = fix_numbers("move it up and right by 100 px", pred(tx=0.2, ty=-0.25), T)
    assert np.allclose(o[:2], np.array([0.7071, -0.7071]) * 100 / 512, atol=1e-3)

def test_scale_factor_and_inverse_for_shrink():
    assert np.isclose(fix_numbers("make it 0.7x", pred(logs=-0.3), S)[4], np.log(0.7)) # type: ignore
    assert np.isclose(fix_numbers("shrink it by 2 times", pred(logs=-0.5), S)[4], np.log(0.5)) # type: ignore
    assert np.isclose(fix_numbers("enlarge it by 2 times", pred(logs=0.5), S)[4], np.log(2)) # type: ignore


def test_scale_percent_by_vs_to():
    assert np.isclose(fix_numbers("reduce it by 30%", pred(logs=-0.2), S)[4], np.log(0.7)) # type: ignore
    assert np.isclose(fix_numbers("reduce it to 30%", pred(logs=-0.2), S)[4], np.log(0.3)) # type: ignore
    assert np.isclose(fix_numbers("enlarge it by 30%", pred(logs=0.2), S)[4], np.log(1.3)) # type: ignore


def test_no_number_leaves_prediction_alone():
    p = pred(tx=0.12, deg=15, logs=0.4) # type: ignore
    assert np.allclose(fix_numbers("make it a bit bigger", p, (False, False, True)), p)


def test_radians_are_converted():
    assert np.isclose(angle(fix_numbers("rotate exactly 1 radian clockwise", pred(deg=-5), R)), -57.2958, atol=1e-3)


def test_unknown_unit_is_not_a_number():
    p = pred(tx=-0.1) # type: ignore
    assert np.allclose(fix_numbers("move left 5 units", p, T), p)


def test_word_factors():
    assert np.isclose(fix_numbers("shrink it to half its size", pred(logs=-0.3), S)[4], np.log(0.5)) # type: ignore
    assert np.isclose(fix_numbers("double the size", pred(logs=0.5), S)[4], np.log(2)) # type: ignore
    assert np.isclose(fix_numbers("make it twice as big", pred(logs=0.5), S)[4], np.log(2)) # type: ignore


def test_percent_goes_to_scale_when_clause_has_a_scale_word():
    o = fix_numbers("scale down 20%", pred(tx=0.1, logs=-0.1), (True, False, True))
    assert np.isclose(o[4], np.log(0.8)) and np.isclose(o[0], 0.1) # type: ignore