import argparse
from inspect import signature

from hermes.typeo.typeo import CustomHelpFormatter, _parse_doc, make_parser


def run_cli():
    parser = argparse.ArgumentParser(
        prog="Nonvex",
        conflict_handler="resolve",
        formatter_class=CustomHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_subparser(fn, name=None):
        description, _ = _parse_doc(fn)
        subparser = subparsers.add_parser(
            name or fn.__name__,
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        make_parser(fn, subparser)
        return subparser

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
        from .search import run_search
    except ImportError:
        pass
    else:
        subparser = add_subparser(run_search, "search")
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
        run_search(**args, args=fn_args)


if __name__ == "__main__":
    run_cli()
