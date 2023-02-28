import torch

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
