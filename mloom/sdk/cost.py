#prices are in USD per 1,000 tokens
#source: OpenAI pricing page
PRICING_TABLE = {
    "gpt-4o":              {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":         {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":         {"input": 0.01,    "output": 0.03},
    "gpt-3.5-turbo":       {"input": 0.0005,  "output": 0.0015},
    "claude-3-opus":       {"input": 0.015,   "output": 0.075},
    "claude-3-sonnet":     {"input": 0.003,   "output": 0.015},
    "claude-3-haiku":      {"input": 0.00025, "output": 0.00125},
}

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Returns total cost in USD. Returns 0.0 if model not found in table.
    
    Example:
        calculate_cost("gpt-4o", 1000, 500)
        # = (1000 * 0.005 / 1000) + (500 * 0.015 / 1000) = 0.005 + 0.0075 = 0.0125
    """
    prices = PRICING_TABLE.get(model_name)
    if not prices:
        return 0.0
    
    input_cost  = (input_tokens  / 1000) * prices["input"]
    output_cost = (output_tokens / 1000) * prices["output"]
    return round(input_cost + output_cost, 8) #is 8 enough?