"""Interpreter start-up shim for the shared remote GPU host.

Place the directory containing this file on PYTHONPATH (it must stay outside
the code tree so the recorded source state is unchanged). It does two things:

* backports ``argparse.BooleanOptionalAction`` on Python 3.8, which the
  fine-tuning scripts use;
* when ``DACP_GPU_MEM_FRACTION`` is set, caps this process's CUDA caching
  allocator on the (single) visible device with
  ``torch.cuda.set_per_process_memory_fraction``.
"""

import argparse
import os

if not hasattr(argparse, "BooleanOptionalAction"):

    class BooleanOptionalAction(argparse.Action):
        """Python 3.9 ``argparse.BooleanOptionalAction`` (``--flag/--no-flag``)."""

        def __init__(
            self,
            option_strings,
            dest,
            default=None,
            type=None,
            choices=None,
            required=False,
            help=None,
            metavar=None,
        ):
            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)
                if option_string.startswith("--"):
                    _option_strings.append("--no-" + option_string[2:])
            if help is not None and default is not None and default is not argparse.SUPPRESS:
                help += " (default: %(default)s)"
            super().__init__(
                option_strings=_option_strings,
                dest=dest,
                nargs=0,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string in self.option_strings:
                setattr(namespace, self.dest, not option_string.startswith("--no-"))

        def format_usage(self):
            return " | ".join(self.option_strings)

    argparse.BooleanOptionalAction = BooleanOptionalAction

_fraction = os.environ.get("DACP_GPU_MEM_FRACTION")
if _fraction:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(_fraction), 0)
