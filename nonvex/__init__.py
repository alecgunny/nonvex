from inspect import signature

from hermes.typeo import typeo


def run_cli():
    kwargs = {}
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

        kwargs["serve"] = serve

    try:
        from .client import main
    except ImportError:
        pass
    else:
        kwargs["search"] = main

    @typeo("Nonvex", **kwargs)
    def nonvex():
        return

    return nonvex()


if __name__ == "__main__":
    run_cli()
