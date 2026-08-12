from multiagent_debate.parsing import (
    majority_vote,
    normalize_number,
    parse_choice_answer,
    parse_numeric_answer,
)


def test_numeric_parser_prefers_boxed_answer() -> None:
    assert parse_numeric_answer("I tried 5, then corrected it to \\boxed{-1,250.00}.") == "-1250"


def test_numeric_parser_supports_gsm_marker_and_last_number() -> None:
    assert parse_numeric_answer("work\n#### 72") == "72"
    assert parse_numeric_answer("The answer is 3.500") == "3.5"
    assert normalize_number("1,000.000") == "1000"


def test_choice_parser_uses_explicit_final_answer() -> None:
    assert parse_choice_answer("A is tempting, but the final answer is c") == "C"
    assert parse_choice_answer("Reasoning complete. (D)") == "D"


def test_majority_vote_and_tie_fallback() -> None:
    assert majority_vote(["A", "B", "A"], fallback="A") == ("A", False)
    assert majority_vote(["A", "B"], fallback="A") == ("A", True)
    assert majority_vote([None, None], fallback=None) == (None, True)

