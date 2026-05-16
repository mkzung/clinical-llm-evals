"""clinical-llm-evals: guideline-cited evaluations for clinical LLMs.

Public API:
    EvalCase         -- pydantic model for a single test case
    EvalResult       -- outcome of running a case
    EvalRunner       -- loads YAML files from a directory and runs them
    AsyncEvalRunner  -- async runner for AsyncLLMClient implementations
    LLMClient        -- Protocol that any sync model adapter must satisfy
    AsyncLLMClient   -- Protocol for async model adapters
    register_scorer  -- decorator for adding custom scoring strategies
    get_reporter     -- factory returning Text/JSON/Markdown/HTML reporters
"""

from clinical_llm_evals.core import (
    AsyncEvalRunner,
    AsyncLLMClient,
    EvalCase,
    EvalResult,
    EvalRunner,
)
from clinical_llm_evals.llm import (
    AnthropicClient,
    LLMClient,
    MockClient,
    OpenAIClient,
)
from clinical_llm_evals.registry import (
    ScorerContext,
    available_scorers,
    get_scorer,
    register_scorer,
)
from clinical_llm_evals.report import get_reporter

__all__ = [
    "AnthropicClient",
    "AsyncEvalRunner",
    "AsyncLLMClient",
    "EvalCase",
    "EvalResult",
    "EvalRunner",
    "LLMClient",
    "MockClient",
    "OpenAIClient",
    "ScorerContext",
    "available_scorers",
    "get_reporter",
    "get_scorer",
    "register_scorer",
]

__version__ = "0.1.0"
