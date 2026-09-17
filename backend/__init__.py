# backend/__init__.py
"""
The HTTP layer.

This package is the web front door and nothing else. It owns request parsing, response
shapes, background execution and error translation - and it owns no product logic at all.
Every endpoint here is a thin wrapper that validates input, calls into
:mod:`multi_agent_generator`, and renders the result as JSON.

That boundary is deliberate and worth defending. The brief asks for both a CLI and a web
app, and the fastest way to end up with two subtly different products is to let the API grow
its own copy of "how to generate a project". So the rule is: if a function here contains a
decision the CLI would also need to make, it is in the wrong package and belongs in
``multi_agent_generator.core``.

Layout::

    app.py        the FastAPI application: middleware, error handlers, router wiring
    config.py     process-wide objects (settings, storage, service) with one owner
    deps.py       FastAPI dependencies - the only place routes obtain those objects
    schemas.py    request and response models
    errors.py     AppError -> HTTP response, and the catch-all for everything else
    api/          one router per resource
    services/     orchestration too web-specific for core (background runs, zip export)

Run it with::

    uvicorn backend.app:app --reload

or ``multi-agent-generator serve``, which is the same thing with the settings applied.
"""
from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """
    Build the FastAPI application.

    Imported lazily so that ``import backend`` does not require FastAPI to be installed -
    the CLI imports this package to find the serve command, and someone who only uses the
    CLI should not need the web extra.
    """
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
