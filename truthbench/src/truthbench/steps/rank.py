import json
import re
from json import JSONDecodeError
from typing import Dict, Any

from truthbench.pipeline import Step, LLM


class RankFactualDataStep(Step):
    """
    Ranks factual items by order of importance.
    """

    PROMPT = (
        "Output the indexes of terms in square brackets [ ] from the text between triple backticks ``` "
        "by terms that shape what the text is about, who it involves, consequences, hard numbers, dates, and facts. "
        "Downrank marked terms that are vague references, general connectors, or dependent on other terms in square "
        "brackets. You are given a free space to decide your ranking strategy between the tags <thinking></thinking>\n"
        "\n"
        "Example:\n"
        "\n"
        "```\n"
        "Recent studies have shown [a correlation:0] between [social media use:1] and [increased anxiety:2] among "
        "[teenagers:3]. Although some researchers argue that online interaction can promote [social connection:4], "
        "others warn about [its impact:5] on [self-esteem:6] and [sleep patterns:7]. Debates intensified after a "
        "[whistleblower:8] revealed internal data from [a major tech company:9] indicating [awareness:10] of "
        "[these risks:11].\n"
        "```\n"
        "<thinking>\n"
        "The main terms are [social media use:1], [teenagers:3], [a major tech company:9]..."
        "Did I forget any missing terms?"
        "..."
        "</thinking>\n"
        "OUTPUT: [1, 3, 9, 2, 8, 6, 7, 4, 10, 0, 11, 5]"
    )

    def __init__(self, llm: LLM, max_retries: int = 8):
        self._llm = llm
        self._max_retries = max_retries
        super().__init__(
            required_fields=frozenset({"with_brackets", "raw_factual_data"}),
            counters=frozenset({"ranked_factual_data", "index_ranking_error", "ranking_factual_data_error"})
        )

    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        if (not sample["with_brackets"] or
                not sample["raw_factual_data"] or
                len(sample["raw_factual_data"]) <= 0):
            sample["ranked_factual_data"] = None
            return

        text = sample["with_brackets"]["A0"]

        terms = re.findall(r'\[([^]]+)]', text)
        for idx, term in enumerate(terms):
            text = text.replace(f'[{term}]', f'[{term}:{idx}]', 1)

        prompt = f"{RankFactualDataStep.PROMPT}\n\nNow it's your turn.\n\n```\n{text}\n```\n"

        for _ in range(self._max_retries):
            llm_judgement = self._llm.query(messages=[{"role": "user", "content": prompt}])

            if "OUTPUT:" not in llm_judgement:
                continue

            value = llm_judgement.split("OUTPUT:")

            if len(value) != 2:
                continue

            _, ranks_str = value

            try:
                ranks = json.loads(ranks_str.strip())
            except JSONDecodeError:
                tracker["json_parse_ranking_error"] += 1
                continue

            if sorted(ranks) == list(range(len(sample["raw_factual_data"]))):
                terms = [sample["raw_factual_data"][i] for i in ranks]
                sample["ranked_factual_data"] = terms
                return

            tracker["index_ranking_error"] += 1

        tracker["ranking_factual_data_error"] += 1
