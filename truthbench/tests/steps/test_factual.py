import re
import unittest
from typing import Callable

import pytest
import spacy.lang.en

from truthbench.steps.factual import NounAdverbFactualChunker, FactualDataStep, FactualChunker


class TestNounAdverbFactualChunker:

    @pytest.mark.parametrize("sentence,expected", [
        (
                "The ozone layer protects the Earth by absorbing harmful ultraviolet radiation from the Sun. It is "
                "primarily found in the stratosphere, a layer of the atmosphere. Concerns about ozone depletion rose "
                "in the 1980s after the discovery of a hole over Antarctica.",
                "The ozone layer protects [the Earth] by absorbing [harmful ultraviolet radiation] from [the Sun]. "
                "It is [primarily] found in [the stratosphere], a layer of the atmosphere. Concerns about ozone "
                "depletion rose in [the 1980s] after [the discovery] of [a hole] over [Antarctica]."
        ),
        (
                "Human activities that elevate the levels of greenhouse gases in the atmosphere are the main drivers "
                "of climate change.",
                "Human activities that elevate [the levels] of [greenhouse gases] in [the atmosphere] are "
                "[the main drivers] of [climate change]."
        ),
        (
                "Castlereagh wounded Canning in the leg, and the incident led to the collapse of the Portland "
                "government and the advancement of Spencer Perceval as the new Prime Minister.",
                "Castlereagh wounded Canning in [the leg], and the incident led to [the collapse] of "
                "[the Portland government] and [the advancement] of [Spencer Perceval] as [the new Prime Minister]."
        ),
        (
                "Renewable energy sources are those that are naturally replenished in short timeframes, including "
                "solar, wind, hydroelectric, geothermal, and biomass energy.",
                "Renewable energy sources are those that are [naturally] replenished in [short timeframes], "
                "including [solar], [wind], [hydroelectric], [geothermal], and [biomass energy]."
        ),
        (
                "The quick brown fox jumps over a box.",
                "The quick brown fox jumps over [a box]."
        ),
        (
                "Human activities are the main drivers of climate change, as they elevate the levels of greenhouse "
                "gases in the atmosphere.",
                "Human activities are [the main drivers] of [climate change], as they elevate [the levels] of "
                "[greenhouse gases] in [the atmosphere]."
        ),
        (
                "Photosynthesis is the mechanism through which plants transform light energy into chemical energy "
                "stored as glucose. This process takes place in the chloroplasts of plant cells, where chlorophyll "
                "captures sunlight. It involves using carbon dioxide from the atmosphere and water from the soil, "
                "resulting in the production of glucose and oxygen as byproducts. The complete chemical equation is: "
                "6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂.",
                "Photosynthesis is [the mechanism] through which plants transform [light energy] into [chemical energy]"
                " stored as [glucose]. This process takes [place] in [the chloroplasts] of [plant cells], where "
                "chlorophyll captures [sunlight]. It involves using [carbon dioxide] from [the atmosphere] and [water] "
                "from [the soil], resulting in [the production] of [glucose] and [oxygen] as [byproducts]. The "
                "complete chemical equation is: [6CO₂ + 6H₂O + light energy] → [C₆H₁₂O₆] + [6O₂.]"
        ),
        (
                "Apple is considering at purchasing an English start-up for 1 billion dollars tomorrow at mid day.",
                "Apple is considering at purchasing [an English start-up] for [1 billion] dollars [tomorrow] at "
                "[mid day]."
        ),
        (
                "Apple is considering at purchasing an English start-up for 1 billion dollars tomorrow morning.",
                "Apple is considering at purchasing [an English start-up] for [1 billion] dollars [tomorrow morning]."
        ),
        (
                "Additionally, deforestation, industrial activities, and farming methods emit substantial quantities "
                "of carbon dioxide, methane, and nitrous oxide, which retain heat in the atmosphere and result in "
                "global warming.",
                "[Additionally], deforestation, industrial activities, and farming methods emit [substantial "
                "quantities] of [carbon dioxide], [methane], and [nitrous oxide], which retain [heat] in [the "
                "atmosphere] and result in [global warming]."
        ),
        (
                "The greatest factor is the combustion of fossil fuels—such as coal, oil, and natural gas—used for "
                "energy and transportation.",
                "The greatest factor is [the combustion] of [fossil fuels]—[such] as [coal], [oil], and "
                "[natural gas]—used for [energy] and [transportation]."
        ),
        (
                "After the storm passed, the volunteers began clearing debris from the roads.",
                "After the storm passed, the volunteers began clearing [debris] from [the roads]."
        ),
        (
                "The machine learning algorithm outperformed traditional models.",
                "The machine learning algorithm outperformed [traditional models]."
        ),
        (
                "When we study hard, we usually do well.",
                "When we study [hard], we [usually] do [well]."
        ),
        (
                "An English start-up is being acquired by Apple for 1 billion dollars tomorrow morning.",
                "An English start-up is being acquired by [Apple] for [1 billion] dollars [tomorrow morning]."
        ),
        (
                "The Castlereagh–Canning duel was a pistol duel fought on September 21, 1809, between the British "
                "Minister of War, Viscount Castlereagh, and Foreign Secretary, George Canning, at Putney Heath. The "
                "reasons for the duel were the rivalry between the two politicians and numerous disagreements between "
                "them over the conduct of the war against Napoleonic France in 1808 and 1809. Castlereagh wounded "
                "Canning in the leg, and the incident led to the collapse of the Portland government and the "
                "advancement of Spencer Perceval as the new Prime Minister. Castlereagh and Canning, meanwhile, "
                "spent several years on the backbenches, absent from any government responsibility.",
                "The Castlereagh–Canning duel was [a pistol duel] fought on [September] [21], [1809], between "
                "[the British Minister] of [War], [Viscount Castlereagh], and [Foreign Secretary], [George Canning], "
                "at [Putney Heath]. The reasons for the duel were [the rivalry] between the [two] politicians and "
                "[numerous disagreements] between them over [the conduct] of [the war] against [Napoleonic France] "
                "in [1808] and [1809]. Castlereagh wounded Canning in [the leg], and the incident led to "
                "[the collapse] of [the Portland government] and [the advancement] of [Spencer Perceval] as [the new "
                "Prime Minister]. Castlereagh and Canning, [meanwhile], spent [several years] on [the backbenches], "
                "absent from [any government responsibility]."
        ),
        (
                "The government announced the new policy in 2021 with confidence.",
                "The government announced [the new policy] in [2021] with [confidence]."
        ),
    ])
    def test_factual_extraction(self, sentence: str, expected: str):
        nlp = spacy.load("en_core_web_sm")
        chunker = NounAdverbFactualChunker(nlp)

        tagged = chunker.tag(sentence)

        assert tagged == expected


class DummyChunker(FactualChunker):

    def __init__(self, transformation: Callable[[str], str] = lambda s: s):
        self.transformation = transformation

    def tag(self, sentence: str) -> str:
        return self.transformation(sentence)


@pytest.fixture
def tracker():
    return {"find_factual_data_error": 0}


def test_missing_a0_key(tracker):
    sample = {"answers": {"A1": "This will be ignored."}}
    step = FactualDataStep(chunker=DummyChunker())

    step.step(sample, tracker)

    assert sample["with_brackets"] is None
    assert sample["raw_factual_data"] is None
    assert tracker["find_factual_data_error"] == 0


def test_no_matches_in_chunker_output(tracker):
    sample = {"answers": {"A0": "Nothing bracketed here."}}
    step = FactualDataStep(chunker=DummyChunker())

    step.step(sample, tracker)

    assert sample["with_brackets"]["A0"] == "Nothing bracketed here."
    assert sample["raw_factual_data"] is None
    assert tracker["find_factual_data_error"] == 1


def test_successful_span_extraction(tracker):
    sample = {"answers": {"A0": "I visited Paris in 2021."}}
    step = FactualDataStep(chunker=DummyChunker(lambda s: s.replace("Paris", "[Paris]").replace("2021", "[2021]")))

    step.step(sample, tracker)

    assert sample["with_brackets"]["A0"] == "I visited [Paris] in [2021]."
    assert sample["raw_factual_data"] == ["Paris", "2021"]
    assert tracker["find_factual_data_error"] == 0


@pytest.mark.parametrize(
    "sample, error",
    [
        (
                {"x": {"A0": "...", "A1": "..."}},  # wrong key
                r"FactualDataStep requires ['answers'], but some are missing from the sample: ['answers']. Check "
                r"pipeline dependencies before proceeding."
        ),
    ]
)
def test_validate(sample, error):
    step = FactualDataStep(DummyChunker())

    with pytest.raises(ValueError, match=re.escape(error)):
        step.validate(sample)


if __name__ == "__main__":
    unittest.main()
