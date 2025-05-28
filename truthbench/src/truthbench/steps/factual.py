import re
from typing import Union, Iterator, List, Tuple, Protocol, Dict, Optional, TypeVar, Generic

from spacy import Language, Errors
from spacy.symbols import NOUN, PROPN, ADV, ADJ, amod, NUM
from spacy.tokens import Doc, Span

from truthbench.pipeline import Step


class SupportsFactualData(Protocol):
    answers: Dict[str, str] = {}
    with_brackets: Dict[str, str] = {}
    raw_factual_data: Optional[List[str]] = []


T = TypeVar('T', bound=SupportsFactualData)


class FactualDataStep(Step, Generic[T]):

    def __init__(self, nlp: Language):
        self.nlp = nlp
        super().__init__(required_fields=("answers", "with_brackets"))

    @classmethod
    def span_boxes(cls, doclike: Union[Doc, Span]) -> Iterator[Span]:
        """
        Detect base noun phrases in the object and adverbs from a dependency parse.
        """
        labels = [
            "oprd",
            "dobj",
            "advmod",
            "amod",
            "npadvmod",
            "pcomp",
            "pobj",
            "dative",
            "appos",
            "attr",
            "ROOT",
        ]

        doc = doclike.doc  # Ensure works on both Doc and Span.

        if not doc.has_annotation("DEP"):
            raise ValueError(Errors.E029)

        np_deps = [doc.vocab.strings.add(label) for label in labels]
        conj = doc.vocab.strings.add("conj")
        prev_end = -1

        # Collect subject heads within the doclike (Span or Doc)
        subject_heads = [token for token in doclike if token.dep_ in {'nsubj', 'nsubjpass'}]

        # Collect indices of all tokens in their subtrees
        subject_indices = set()
        for head in subject_heads:
            # Add all tokens in the subject head's subtree
            head_subtree = {t.i for t in head.subtree}
            subject_indices.update(head_subtree)

            # Subtract tokens in relative clauses (relcl) attached to the subject head
            for child in head.children:
                if child.dep_ == "relcl":
                    relcl_subtree = {t.i for t in child.subtree}
                    subject_indices.difference_update(relcl_subtree)

        for i, word in enumerate(doclike):
            if word.pos not in (NOUN, PROPN, ADV, ADJ, NUM):
                continue

            # Skip if part of the subject
            if word.i in subject_indices:
                continue

            if word.pos == ADJ and word.dep == amod and word.head.pos in (NOUN, PROPN):
                continue

            # Prevent nested chunks from being produced
            if word.left_edge.i <= prev_end:
                continue

            if word.dep in np_deps or (word.pos == NUM and word.dep_ in ("nummod", "appos", "attr")):
                prev_end = word.i
                yield doc[word.left_edge.i:word.i + 1]
            elif word.dep == conj:
                head = word.head

                while head.dep == conj and head.head.i < head.i:
                    head = head.head

                # If the head is an NP, and we're coordinated to it, we're an NP
                if head.dep in np_deps:
                    prev_end = word.i
                    yield doc[word.left_edge.i:word.i + 1]

    @classmethod
    def overlaps(cls, idx: List[Tuple[int, int]]) -> bool:
        sorted_intervals = sorted(idx)
        return any(
            current_end > next_start
            for (_, current_end), (next_start, _) in zip(sorted_intervals, sorted_intervals[1:])
        )

    def tag_predicate_nouns_and_adverbs(self, sentence: str):
        doc = self.nlp(sentence)

        idx = []
        for box in FactualDataStep.span_boxes(doc):
            idx.append((min(b.idx for b in box), max(b.idx + len(b) for b in box)))

        idx.sort(reverse=True)

        assert not FactualDataStep.overlaps(idx), f"Something went wrong... Overlapping indexes for `{sentence}`"

        boxed_sentence = sentence
        for start, end in idx:
            boxed_sentence = boxed_sentence[:start] + "[" + boxed_sentence[start:end] + "]" + boxed_sentence[end:]

        return boxed_sentence

    def step(self, sample: SupportsFactualData, tracker: Tracker) -> None:
        assert "A0" in sample.answers.keys()

        response_text = self.tag_predicate_nouns_and_adverbs(sample.answers["A0"])
        sample.with_brackets["A0"] = response_text

        matches = re.findall(r"\[(.*?)]", response_text)
        if not matches:
            tracker.find_factual_data_error += 1
            return

        sample.raw_factual_data = matches
