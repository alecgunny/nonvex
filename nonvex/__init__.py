import argparse
from inspect import signature

from hermes.typeo.typeo import make_parser


def run_cli():
    parser = argparse.ArgumentParser(
        prog="Nonvex",
        conflict_handler="resolve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_subparser(fn):
        subparser = subparsers.add_parser(
            fn.__name__,
            description=fn.__doc__.split("Args:")[0],
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        return make_parser(fn, None, subparser, None)

    try:
        from .app import create_app
    except ImportError:
        pass
    else:

        def serve(**kwargs):
            app = create_app(**kwargs)
            app.run()

        serve.__signature__ = signature(create_app)
        serve.__doc__ = create_app.__doc__
        add_subparser(serve)

    try:
        from .client import search
    except ImportError:
        pass
    else:
        subparser, _ = add_subparser(search)
        for action in subparser._actions:
            if action.option_strings[0] == "--executable":
                action.option_strings = []

    args, fn_args = parser.parse_known_args()
    args = vars(args)
    command = args.pop("command")
    if command == "serve":
        if len(fn_args) > 0:
            raise parser.ArgumentError("Unknown arguments {}".format(fn_args))
        serve(**args)
    else:
        args.pop("args")
        search(**args, args=fn_args)


if __name__ == "__main__":
    run_cli()
