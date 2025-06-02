import abc
import re
from typing import Union, Iterator, List, Tuple, Dict, Any

from spacy import Language, Errors
from spacy.symbols import NOUN, PROPN, ADV, ADJ, amod, NUM
from spacy.tokens import Doc, Span

from truthbench.pipeline import Step


class FactualChunker(abc.ABC):
    """
    Extract factual components of a sentence.
    """

    @abc.abstractmethod
    def tag(self, sentence: str) -> str:
        ...


class NounAdverbFactualChunker(FactualChunker):
    """
     Implements a rule-based factual chunker that identifies textual spans likely to carry factual content
     within a sentence, focusing on noun phrases and adverbial modifiers.

     This chunker uses syntactic dependency parsing to detect spans such as:
       - Direct objects, attributes, and complements of the main verb.
       - Adverbial modifiers and prepositional phrases expressing circumstantial details.
       - Appositional phrases and descriptive noun modifiers.
       - Numerical expressions excluding those in the subject.

     To avoid altering the core meaning of sentences, it excludes spans that belong to grammatical subjects.
     For noun phrase subjects containing embedded relative clauses, only the head noun is excluded,
     allowing modifiers in relative clauses to be eligible.

     Coordination is handled by propagating eligibility from a conjunct to its siblings if the head
     satisfies the criteria. The chunker also suppresses nested or overlapping spans to produce a clean,
     non-redundant set of factual candidates.

     The output is a bracketed string marking identified factual spans, forming an intermediate
     representation for downstream filtering, ranking, or perturbation.

     Example:
         Input:  "The government announced the new policy in 2021 with confidence."
         Output: "The government announced [the new policy] in [2021] with [confidence]."

     Methods:
         - tag(sentence: str) -> str:
             Returns the input sentence with factual spans bracketed.

     Notes:
         - Requires a syntactic dependency parse (e.g., from spaCy).
         - Focuses on spans relevant for factual content modification.
         - Does not modify spans related to sentence subjects to prevent meaning distortion.
     """

    def __init__(self, nlp: Language):
        self._nlp = nlp

    def span_boxes(self, doclike: Union[Doc, Span]) -> Iterator[Span]:
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

    def overlaps(self, idx: List[Tuple[int, int]]) -> bool:
        sorted_intervals = sorted(idx)
        return any(
            current_end > next_start
            for (_, current_end), (next_start, _) in zip(sorted_intervals, sorted_intervals[1:])
        )

    def tag(self, sentence: str) -> str:
        doc = self._nlp(sentence)

        idx = []
        for box in self.span_boxes(doc):
            idx.append((min(b.idx for b in box), max(b.idx + len(b) for b in box)))

        idx.sort(reverse=True)

        assert not self.overlaps(idx), \
            f"Something went wrong... Overlapping indexes for `{sentence}`"

        boxed_sentence = sentence
        for start, end in idx:
            boxed_sentence = boxed_sentence[:start] + "[" + boxed_sentence[start:end] + "]" + boxed_sentence[end:]

        return boxed_sentence


class FactualDataStep(Step):
    """
    Step that identifies factual data spans within an answer text by leveraging a
    provided FactualChunker implementation. It marks spans in the answer likely to
    contain factual content, brackets them, and extracts these spans for downstream processing.

    Attributes:
        - chunker (FactualChunker): An instance responsible for tagging factual spans in sentences.

    Expected Sample Fields:
        - answers (Dict[str, str]): A dictionary containing answer texts keyed by identifiers
          (e.g., "A0"). This step processes the text under the "A0" key.

    Modifies:
        - sample["with_brackets"] (Dict[str, str] or None): Adds a dictionary mapping answer keys
          to bracketed strings marking factual spans. Set to None if input is missing or invalid.
        - sample["raw_factual_data"] (List[str] or None): Extracted factual spans as a list of strings.
          Set to None if no factual spans are found.

    Counter:
        - find_factual_data_error: Incremented when no factual spans are detected in the answer text.

    Notes:
        - Relies on the injected FactualChunker to perform the actual span identification and tagging.
        - If "answers" is missing or does not contain the key "A0", no processing occurs and relevant
          fields are set to None.
        - Extracted spans are obtained by regex matching bracketed sections in the tagged text.

    Example:
        sample = {
            "answers": {"A0": "The government announced the new policy in 2021 with confidence."}
        }
        tracker = {"find_factual_data_error": 0}

        chunker = NounAdverbFactualChunker(nlp)
        step = FactualDataStep(chunker)
        step.step(sample, tracker)

        # After processing:
        # sample["with_brackets"]["A0"] will be:
        # "The government announced [the new policy] in [2021] with [confidence]."
        # sample["raw_factual_data"] will be:
        # ["the new policy", "2021", "confidence"]
        # tracker["find_factual_data_error"] remains 0 because factual spans were found.
    """

    def __init__(self, chunker: FactualChunker):
        self._chunker = chunker
        super().__init__(
            required_fields=frozenset({"answers"}),
            counters=frozenset({"find_factual_data_error"})
        )

    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        if not sample["answers"] or "A0" not in sample["answers"].keys():
            sample["with_brackets"] = None
            sample["raw_factual_data"] = None
            return

        sample["with_brackets"] = {}
        response_text = self._chunker.tag(sample["answers"]["A0"])
        sample["with_brackets"]["A0"] = response_text

        matches = re.findall(r"\[(.*?)]", response_text)
        if not matches:
            tracker["find_factual_data_error"] += 1
            sample["raw_factual_data"] = None
            return

        sample["raw_factual_data"] = matches
