"""
Console CRUD routers, one module per resource.

Kept in its own package, outside the orchestration import graph, for the reason
`store.py` exists at all: these modules import FastAPI and Pydantic, and the state
machine must stay importable with nothing installed but pytest. `avatar.orchestrator`
and friends never import from here, and `test_boundaries.py` is what keeps that true.
"""
