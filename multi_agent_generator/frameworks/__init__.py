from .crewai_generator import create_crewai_code
from .crewai_flow_generator import create_crewai_flow_code
from .langgraph_generator import create_langgraph_code
from .react_generator import create_react_code, create_react_lcel_code
from .agno_generator import create_agno_code

#: Maps a framework name to the function that generates its code. Every generator takes
#: ``(config, provider="openai", model=None)``, so callers can dispatch without a
#: hand-written if/elif chain.
GENERATORS = {
    'crewai': create_crewai_code,
    'crewai-flow': create_crewai_flow_code,
    'langgraph': create_langgraph_code,
    'react': create_react_code,
    'react-lcel': create_react_lcel_code,
    'agno': create_agno_code,
}

#: Framework names in a stable order, suitable for argparse choices.
FRAMEWORKS = list(GENERATORS)


def generate_code(config, framework, provider='openai', model=None):
    """
    Generate agent code for ``framework``.

    Args:
        config: The agent configuration dictionary.
        framework: One of :data:`FRAMEWORKS`.
        provider: LLM provider backing the generated agents.
        model: Optional explicit model id overriding the provider default.

    Returns:
        Generated Python code as a string.
    """
    try:
        generator = GENERATORS[framework]
    except KeyError:
        known = ', '.join(FRAMEWORKS)
        raise ValueError(
            f'Unsupported framework {framework!r}. Supported frameworks: {known}'
        ) from None
    return generator(config, provider=provider, model=model)


__all__ = [
    'create_crewai_code',
    'create_crewai_flow_code',
    'create_langgraph_code',
    'create_react_code',
    'create_react_lcel_code',
    'create_agno_code',
    'generate_code',
    'GENERATORS',
    'FRAMEWORKS',
]
