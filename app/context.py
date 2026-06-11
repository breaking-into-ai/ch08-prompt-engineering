"""Context engineering: budget every token that reaches the model."""
import tiktoken

CONTEXT_BUDGET = {
    "system_prompt": 800,
    "few_shot_examples": 600,
    "user_preferences": 400,
    "conversation_history": 2000,
    "current_query": 200,
    "output_reserve": 1000,
}
ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def trim_to_budget(text: str, budget: int) -> str:
    tokens = ENC.encode(text)
    return ENC.decode(tokens[-budget:]) if len(tokens) > budget else text


def manage_history(full_history, summarize_fn, max_recent=4):
    """Keep recent turns verbatim; compress older turns to a preference summary."""
    if len(full_history) <= max_recent:
        return full_history
    summary = summarize_fn(full_history[:-max_recent])
    return [{"role": "system", "content": f"User preference summary: {summary}"}] \
        + full_history[-max_recent:]


class ContextPipeline:
    def __init__(self, system_prompt, token_budget=8000):
        self.system_prompt = system_prompt
        self.budget = token_budget

    def build_messages(self, query, preferences="", history_summary=""):
        from .prompts.recommend_v3 import USER_TEMPLATE
        user = USER_TEMPLATE.format(preferences=trim_to_budget(preferences, 400),
                                    history_summary=trim_to_budget(history_summary, 2000),
                                    query=trim_to_budget(query, 200))
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user}]
