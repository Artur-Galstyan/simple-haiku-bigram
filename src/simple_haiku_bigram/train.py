"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = simple_haiku_bigram.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""
import argparse
import logging
import os
import sys

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax import jit
from tinyshakespeareloader.hamlet import get_data
from torch.utils.data import DataLoader

from simple_haiku_bigram import __version__, bigram_model

try:
    import cPickle as pickle
except ImportError:
    import pickle


__author__ = "Artur A. Galstyan"
__copyright__ = "Artur A. Galstyan"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

_model = None
# _params = None


def init_model():
    data = get_data()
    vocab_size = data["vocabulary_size"]
    model = hk.transform(lambda x: bigram_model.SimpleBigram(vocab_size)(x))
    return model


def net(params, batch, rng_key):
    # init global _model
    global _model  # pylint: disable=global-statement
    if _model is None:
        _model = init_model()
    return _model.apply(params=params, x=batch, rng=next(rng_key))


def loss_fn(
    params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray
) -> jnp.ndarray:
    rng_key = hk.PRNGSequence(42)
    output = net(params, batch, rng_key)
    # output = model.apply(params=params, x=batch, rng=next(rng_key))
    loss_value = optax.softmax_cross_entropy_with_integer_labels(output, labels)
    return loss_value.mean()


def get_weights_folder():
    # get the absolute path of the weights folder
    weights_folder = os.path.join(os.path.dirname(__file__), "weights")
    # create the weights folder if it doesn't exist
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
    return weights_folder


def fit(
    params: optax.Params,
    optim: optax.GradientTransformation,
    train_dataloader: DataLoader,
):
    opt_state = optim.init(params)

    @jit
    def step(params, opt_state, batch, labels):
        loss_value, grads = jax.value_and_grad(loss_fn)(params, batch, labels)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i, (x, y) in enumerate(train_dataloader):
        with jax.checking_leaks():
            params, opt_state, loss_value = step(
                params, opt_state, x.numpy(), y.numpy()
            )
        if i % (len(train_dataloader) // 10) == 0:
            _logger.info(f"step={(i / len(train_dataloader)) * 100}%, {loss_value=}")

    weights_folder = get_weights_folder()
    # Save the params object to the weights folder
    _logger.info("Saving params.pkl file")
    with open(weights_folder + "/params.pkl", "wb") as f:
        pickle.dump(params, f)

    return params


def load_params():
    rng_key = hk.PRNGSequence(42)
    # import params from the weights folder if it exists
    try:
        weights_folder = get_weights_folder()
        with open(weights_folder + "/params.pkl", "rb") as f:
            params = pickle.load(f)
    except FileNotFoundError:
        _logger.info("No params.pkl file found, initializing new params")
        train_dataloader = get_data()["train_dataloader"]
        dummy_data, _ = next(iter(train_dataloader))
        params = _model.init(rng=next(rng_key), x=dummy_data.numpy())
    return params


def train():
    global _model  # pylint: disable=global-statement
    _model = init_model()
    params = load_params()

    train_dataloader = get_data()["train_dataloader"]
    learning_rate = 1e-3
    optim = optax.adamw(learning_rate=learning_rate)
    params = fit(params, optim, train_dataloader)
    return params


def blabber(max_new_tokens=100):
    global _model  # pylint: disable=global-statement
    _model = init_model()
    params = load_params()
    text = bigram_model.generate_new_tokens(params, _model, max_new_tokens)
    decode = get_data()["decode"]
    text = decode(text[0].tolist())

    return text


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Simple Haiku Bigram Model")
    parser.add_argument(
        "--version",
        action="version",
        version=f"simple-haiku-bigram {__version__}",
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="skip training step",
        dest="no_training",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    # get log level from args (if present) and setup setup_logging
    args = parse_args(args)
    if args.loglevel is None:
        args.loglevel = logging.WARNING
    setup_logging(args.loglevel)
    if not args.no_training:
        train()
    blabber()


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m simple_haiku_bigram.skeleton 42
    #
    run()
