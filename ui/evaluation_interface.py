import argparse
import difflib
import html
import json
import pathlib
import random
from typing import List, Dict

import gradio as gr
import pydantic

# Set up argument parser
parser = argparse.ArgumentParser(description='Launch the evaluation interface')
parser.add_argument('input_file', type=pathlib.Path, help='Path to the evaluation dataset JSON file')
parser.add_argument('output_file', type=pathlib.Path, help='Path to the assessment records')

# Parse the arguments
args = parser.parse_args()


class AssessmentItem(pydantic.BaseModel):
    id: int
    question: str
    ground_truth: str
    answers: Dict[str, Dict[str, str]]


class AssessmentDataset(pydantic.BaseModel):
    questions: List[AssessmentItem]


# Load the dataset
with open(args.input_file, "r") as f:
    data = AssessmentDataset.model_validate_json(f.read())


class EvaluationItem:
    def __init__(self, item: AssessmentItem, human_on_left: bool):
        self.item = item
        self.human_on_left = human_on_left


inputs: List[EvaluationItem] = []


def word_diff(prev: str, current: str) -> str:
    prev_words = prev.split()
    curr_words = current.split()

    diff = difflib.ndiff(prev_words, curr_words)
    result = []

    for word in diff:
        tag = word[:2]
        text = html.escape(word[2:])
        if tag == "- ":
            result.append(f"<span style='color:red;text-decoration:line-through;'>{text}</span>")
        elif tag == "+ ":
            result.append(f"<span style='color:green;'>{text}</span>")
        elif tag == "  ":
            result.append(f"{text}")
    return " ".join(result)


# Function to evaluate responses
def evaluate_responses(*radio_values):
    results = {}
    i = 0
    for e in inputs:
        question_id = e.item.id
        results[question_id] = {}
        for level in sorted(e.item.answers.keys()):
            result = radio_values[i]
            i += 1
            if result in ("Response 1", "Response 2"):
                if (result == "Response 1" and e.human_on_left) or (result == "Response 2" and not e.human_on_left):
                    result = "Expert"
                else:
                    result = "AI"
            results[question_id].update({level: result})

    # Save results to a JSON file
    with open(args.output_file, "w") as outfile:
        json.dump(results, outfile, indent=4)

    return f"Preferences submitted successfully. Results saved to {args.output_file}."


# Create Gradio interface
def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# 📌 Goals of the Evaluation Procedure")

        gr.Markdown(
            "This evaluation aims to measure how well an automated pipeline can mimic human-crafted answers across "
            "varying levels of factual correctness. In this blind test, you will compare pairs of answers—"
            "one generated by the pipeline and one authored by a human expert—for each error level (**A0** to **A4**)."
        )

        gr.Markdown(
            "**Each level is designed to reflect a different degree of correctness:**\n\n"
            "- **A0**: As accurate and comprehensive as the ground truth.\n"
            "- **A1–A3**: Gradual decline in factual accuracy and coherence.\n"
            "- **A4**: Mostly incorrect, though potentially still plausible at surface level."
        )
        with gr.Accordion("🛠️ Instructions", open=True):
            gr.Markdown("## 🎯 Your task:")

            gr.Markdown(
                "For each level (**A0 to A4**), **select the answer (pipeline or human)** that best corresponds to the "
                "intended degree of correctness. You're not assessing which answer is “better” in isolation, but which "
                "one more appropriately reflects the **target quality level**."
            )

            gr.Markdown("## 📝 Consider the following:")

            gr.Markdown(
                "- Does the selected answer reflect the **intended factual quality** of the level?\n"
                "- Is the answer **too correct or too incorrect** for the target level?\n"
                "- Does one of the answers exhibit subtle errors, misleading phrasing, or hallucinations that better "
                "match the expected degradation?\n"
                "- Does the introduced errors are too obvious?"
            )

            gr.Markdown(
                "**The ultimate goal** is to determine whether the pipeline can generate answers that *faithfully "
                "emulate* the intended quality tiers, as well (or better) than human experts."
            )

        radios: List[gr.Radio] = []
        for question in data.questions:
            question_text = question.question
            ground_truth = question.ground_truth

            human_on_left = random.choice([True, False])
            inputs.append(EvaluationItem(question, human_on_left))
            # Determine the positions for this question
            with gr.Accordion(f"Question: {question_text}", open=False):
                gr.Markdown(f"**Ground Truth:** {ground_truth}")

                a0_ai_response = question.answers["A0"]["ai"]
                a0_human_response = question.answers["A0"]["human"]
                for k, a in sorted(question.answers.items(), key=lambda i: i[0]):
                    level = k
                    human_response, ai_response = a["human"], a["ai"]

                    human_diff_html = word_diff(a0_human_response, human_response)
                    ai_diff_html = word_diff(a0_ai_response, ai_response)

                    with gr.Row():
                        # For Response 1
                        with gr.Column():
                            diff_state_1 = gr.State(False)
                            toggle_button_1 = gr.Button("🔍 Show Diff")
                            plain_1 = gr.Markdown(human_response if human_on_left else ai_response, visible=True)
                            diff_1 = gr.Markdown(human_diff_html if human_on_left else ai_diff_html, visible=False)

                            def toggle_diff_btn_1(show):
                                new_show = not show
                                return [
                                    gr.update(visible=not new_show),  # plain_1
                                    gr.update(visible=new_show),  # diff_1
                                    gr.update(value="🔍 Hide Diff" if new_show else "🔍 Show Diff"),
                                    # toggle_button_1 label
                                    new_show  # update state
                                ]

                            toggle_button_1.click(
                                fn=toggle_diff_btn_1,
                                inputs=[diff_state_1],
                                outputs=[plain_1, diff_1, toggle_button_1, diff_state_1]
                            )

                        with gr.Column():
                            diff_state_2 = gr.State(False)
                            toggle_button_2 = gr.Button("🔍 Show Diff")
                            plain_2 = gr.Markdown(ai_response if human_on_left else human_response, visible=True)
                            diff_2 = gr.Markdown(ai_diff_html if human_on_left else human_diff_html, visible=False)

                            def toggle_diff_btn_2(show):
                                new_show = not show
                                return [
                                    gr.update(visible=not new_show),  # plain_1
                                    gr.update(visible=new_show),  # diff_1
                                    gr.update(value="🔍 Hide Diff" if new_show else "🔍 Show Diff"),
                                    # toggle_button_1 label
                                    new_show  # update state
                                ]

                            toggle_button_2.click(
                                fn=toggle_diff_btn_2,
                                inputs=[diff_state_2],
                                outputs=[plain_2, diff_2, toggle_button_2, diff_state_2]
                            )

                    radios.append(
                        gr.Radio(
                            choices=["Response 1", "Response 2", "Both are good", "Both are bad"],
                            label=f"Level {level}: Which response best aligns to this level?",
                            interactive=True
                        )
                    )

        submit_button = gr.Button("Submit Preferences")
        output_text = gr.Textbox(label="Evaluation Results", interactive=False)

        # Final submission button
        submit_button.click(evaluate_responses, inputs=radios, outputs=[output_text])

    return interface


# Launch the interface
if __name__ == "__main__":
    create_interface().launch()
