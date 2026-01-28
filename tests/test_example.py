from examples.example import sum_squares


def test_sum_squares_small():
    assert sum_squares(10) == sum(i*i for i in range(10))


def test_sum_squares_zero():
    assert sum_squares(0) == 0
