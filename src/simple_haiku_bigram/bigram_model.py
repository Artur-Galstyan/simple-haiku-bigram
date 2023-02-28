import haiku as hk
import jax
import jax.numpy as jnp
from chex import assert_rank


class SimpleBigram(hk.Module):
    def __init__(self, vocab_size) -> None:
        super(SimpleBigram, self).__init__()
        self.vocab_size = vocab_size

        self.embedding_table = hk.Embed(vocab_size=vocab_size, embed_dim=vocab_size)

    def __call__(self, x):
        assert_rank(x, 2)

        logits = self.embedding_table(x)

        assert_rank(logits, 3)
        return logits


def generate_new_tokens(params, model, max_new_tokens=100):
    rng = hk.PRNGSequence(42)
    idx = jnp.zeros(shape=(1, 1), dtype=jnp.int32)
    for _ in range(max_new_tokens):
        output = model.apply(params=params, x=idx, rng=next(rng))
        output = output[:, -1, :]
        next_idx = jax.random.categorical(next(rng), output).reshape(-1, 1)
        idx = jnp.concatenate((idx, next_idx), axis=1)

    return idx
