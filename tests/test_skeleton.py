import pytest
import torch

from simple_haiku_bigram.skeleton import fib, main
from simple_haiku_bigram.train import blabber, train

__author__ = "Artur A. Galstyan"
__copyright__ = "Artur A. Galstyan"
__license__ = "MIT"


def test_blabber():
    max_new_tokens = 100
    text = blabber(max_new_tokens=max_new_tokens)
    assert text is not None
    assert len(text) == max_new_tokens + 1


def test_training():
    torch.manual_seed(0)
    params = train()
    assert params is not None


def test_fib():
    """API Tests"""
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)


def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    main(["7"])
    captured = capsys.readouterr()
    assert "The 7-th Fibonacci number is 13" in captured.out
