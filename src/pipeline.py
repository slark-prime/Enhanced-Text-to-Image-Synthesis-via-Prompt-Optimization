import json
import random
import re
import requests
import os
import time
from diffusers import AutoPipelineForText2Image
import torch
from src.agents import Generator, Scorer, InstructionModifier, GradientCalculator
import heapq
import numpy as np
from collections import defaultdict

# enter api key and org
NEW_INSTRUCTION_NUM = 5
API_KEY = ""
ORG = ""


class SD:
    def __init__(self) -> None:
        self.model_id = "stabilityai/sdxl-turbo"
        self.pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, torch_dtype=torch.float16,
                                                              variant="fp16").to("cuda")


def default_value():
    return [[], 1]


class AgentsPipline:
    def __init__(self, instr_list_length=3) -> None:
        self.generator = Generator(API_KEY, ORG)
        self.scorer = Scorer()
        self.instruction_modifier = InstructionModifier(API_KEY, ORG)
        self.Diffusion_M = SD()
        self.query_to_prompt = {}
        self.instruction_list = {}
        self.instrcution_n = defaultdict(default_value)
        self.gradient_calculator = GradientCalculator(API_KEY, ORG)
        self.b = instr_list_length  # instruction_list length
        self.initial_instruction = "This is the original prompt that you need to carefully refine"
        self.training_p_s = []
        self.final_instruction = ""

    @staticmethod
    def fetch_prompts_from_lexica(query):
        response = requests.get(f"https://lexica.art/api/v1/search?q={query}")
        if response.status_code == 200:
            data = response.json()
            lexica_prompts = [item["prompt"] for item in data["images"]]
            return lexica_prompts
        else:
            print("Failed to fetch prompts from Lexica")
            return []

    @staticmethod
    def fetch_prompts_from_jsonl(query_filter, file_path="lexica_prompts.txt"):
        """
        Fetches prompts from a local .jsonl file, filtering based on the provided query.

        :param file_path: Path to the local .jsonl file.
        :param query_filter: The query to filter prompts by.
        :return: A list of prompts matching the given query.
        """
        prompts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Parse the JSON object from each line
                    obj = json.loads(line.strip())
                    # Check if the object's query matches the filter, and if so, add the prompt to our list
                    if obj.get('query') == query_filter:
                        prompts.append(obj.get('prompt'))
            return prompts
        except Exception as e:
            print(f"Failed to fetch prompts from the local file: {e}")
            return []

    def G_output(self, query, instruction=None):
        if instruction is None:
            instruction = self.initial_instruction

        # print(f"G_output: {instruction}, Prompt to Modify or subject: '{query}' ")
        self.generator.add_instruction(f"{instruction}, Prompt or subject to refine : {query}")
        generated_prompt = self.generator.get_last_run_response()
        return generated_prompt

    def IM_output(self, meg):
        self.instruction_modifier.add_instruction(meg)
        new_instruction = self.instruction_modifier.get_last_run_response()
        # print(f"IM_output: New instruction: {new_instruction}")
        return new_instruction

    def loss_compute_and_track_batch(self, prompt_info):
        """
        Computes the loss for a batch of prompts and tracks the results.

        """

        prompt = prompt_info['prompt']
        instruction = prompt_info.get('instruction')
        source = prompt_info['source']

        image = self.Diffusion_M.pipe(prompt).images[0]
        file_name = f"{prompt_info['query']}_{source}_{instruction[:10]}.png" if source == 'generated' and instruction is not None \
            else f"{prompt_info['query']}_{source}_lexica_{prompt[:5]}.png"
        os.makedirs("./SD_images", exist_ok=True)
        image_path = os.path.join("./SD_images", file_name)
        image.save(image_path)
        loss = self.scorer.get_loss(image_path, prompt)
        print(f" loss: {loss}")

        return loss

    def find_extreme_instruction(self, n=1):
        # """

        # Finds the tracks for the n highest scoring prompts and n lowest scoring instructions.

        # :param n: Number of top and bottom scores to retrieve.
        # :return: A tuple containing lists of the n highest scores and n lowest scores.
        # """

        """
        Locate n lowest batch loss instruction
        """
        if len(self.instruction_list) == 0 or n < 1:
            print("Invalid input or no prompts have been tracked yet.")
            return [], []

        # Sort the tracked prompts by loss
        sorted_instruction = sorted(list(self.instruction_list.items()), key=lambda x: x[1]['batch_loss'])

        # Retrieve the n highest scores (lowest losses) and n lowest scores (highest losses)

        print(f"highest_score_instruction:{sorted_instruction[-1][0]}, score{sorted_instruction[-1][1]['batch_loss']}")
        self.training_p_s.append(sorted_instruction[-1][1]['batch_loss'])
        return sorted_instruction[:n]

    @staticmethod
    def extract_improvement(text):
        """Extracts inferences based on the 'Inference N:' pattern from a text block."""
        pattern = re.compile(r'Improvement \d+: ([^\n]+)')
        inferences = pattern.findall(text)
        return inferences

    def generate_prompts_and_fetch_track(self, instruction_list, query):
        """
      Generate prompts based on the given instructions and a query, track their performance.

      :param instruction_list: List of instructions to use for prompt generation.
      :param query: Query to fetch external prompts for additional variety.
      """

        # Generate prompts using each instruction
        for instruction in list(instruction_list.keys()):
            instruction_dic = {"batch_loss": 0, "querys": {}}
            inst_batch_loss = 0
            for q in query:
                # store generated prompt batch and calculated loss in tuple according to query
                generated_prompt = self.G_output(q, instruction)
                loss_q = self.loss_compute_and_track_batch(
                    {'prompt': generated_prompt, 'instruction': instruction, 'source': 'generated', "query": q})
                self.instrcution_n[instruction][0].append(loss_q)
                inst_batch_loss += loss_q
                prompt_score_tuple = (generated_prompt, loss_q)
                generate_prompts = instruction_dic.get("querys")
                generate_prompts[q] = prompt_score_tuple

                # add into all prompt pool

                if q not in self.query_to_prompt:
                    self.query_to_prompt[q] = []
                self.query_to_prompt[q].append(prompt_score_tuple)

                # fetch prompt from lexico and add the tuple into all prompt pool
                # TODO: here fetch the prompt from lexica

                self.query_to_prompt[q] = sorted(self.query_to_prompt[q], key=lambda x: x[1])

            instruction_dic["batch_loss"] = inst_batch_loss / len(query)
            print(instruction_dic["batch_loss"])
            instruction_list[instruction] = instruction_dic

    def analyze_and_propose(self):
        # Extract highest and lowest scores
        lowest_instructions = self.find_extreme_instruction(n=1)  # Example with n=1

        new_instructions = []
        # Generate new instructions based on analysis
        # Update instructions only for the lowest-scoring prompts with a 'generated' source
        for lowest_instruction in lowest_instructions:
            # generate the highest score and lowest score prompt pair according to the lowest batch score instruction
            highest_score_prompt = []
            lowest_score_prompt_for_instr = []
            lowest_inst_query_prompts_pair = lowest_instruction[1]["querys"].items()
            print(lowest_inst_query_prompts_pair)
            for query, (gener_prompt, loss) in lowest_inst_query_prompts_pair:
                highest_score_prompt_to_query = self.query_to_prompt[query][-1]
                if gener_prompt != highest_score_prompt_to_query[0]:
                    highest_score_prompt.append((query, highest_score_prompt_to_query))
                    lowest_score_prompt_for_instr.append((query, (gener_prompt, loss)))
            original_instruction = lowest_instruction
            if len(lowest_score_prompt_for_instr) == 0:

                analysis_feedback = f"based on the initial instruction '{original_instruction}', generate three different instructions by adding requirments to generated prompt in thel instruction, while ensuring diversity and the added requirment can make the generated prompt more speicific.your output should be:\n Instruction 1:your new generated instr_1 \n Instruction 2:your new generated instr_2 \n Instruction 3:your new generated instr_3 "

                new_instruction_init = self.IM_output(analysis_feedback)
                new_insts_init = re.findall(r'Instruction \d+:(.+)', new_instruction_init)
                # Trim whitespace from each matched content
                new_insts_init = [out_space.strip() for out_space in new_insts_init]
                for instr in new_insts_init:
                    print("analyze_and_propose: Updated Instruction:", instr)
                    self.instruction_list[instr] = {"batch_loss": 0, "querys": {}}
                del self.instruction_list[lowest_instruction[0]]

            else:
                reasons_summary = self.gradient_calculator.analyze_prompts(lowest_instruction[0],
                                                                           lowest_score_prompt_for_instr,
                                                                           highest_score_prompt)

                print(f"analyze_and_propose: Summary of Reasons for Scores:{reasons_summary}")

                improvements = self.extract_improvement(reasons_summary)

                for improvement in improvements:
                    print("analyze_and_propose: improvement:", improvement)
                    # Assuming 'instruction' is available in the prompt_info structure

                    # Preparing feedback for instruction modification
                    analysis_feedback = f"Based on improvemnt suggestion: {improvement}, consider improving the instruction: '{original_instruction}' , while try to avoid mention specific information about the prompt'"
                    new_instruction = self.IM_output(analysis_feedback)
                    print("analyze_and_propose: Updated Instruction:", new_instruction)
                    self.instruction_list[new_instruction] = {"batch_loss": 0, "querys": {}}

                del self.instruction_list[lowest_instruction[0]]

    def maintain_instructions_list(self):
        # Ensure the list is capped at self.b elements
        if len(self.instruction_list) > self.b:
            sampled_items = random.sample(list(self.instruction_list.items()), self.b)
            self.instruction_list = dict(sampled_items)

    def greedy_instruction_select(self):
        top_n_inst = heapq.nlargest(min(self.b, len(self.instruction_list)), self.instruction_list.items(),
                                    key=lambda item: item[1]['batch_loss'])
        self.instruction_list = dict(top_n_inst)

        print(f"top_instr:{self.instruction_list}")

    def ucb(self, c, t):
        def get_u(instruction, t):
            n = self.instrcution_n[instruction][1]
            self.instrcution_n[instruction][1] += 1
            log = np.log(t) / n
            return c * log ** 0.5

        select_dict = {}
        for i in range(min(self.b, len(self.instruction_list))):
            max_ = max(self.instruction_list.keys(), key=lambda x: np.mean(self.instrcution_n[x][0]) + get_u(x, t + i))
            select_dict[max_] = self.instruction_list.pop(max_)

        self.instruction_list = select_dict

    def epsilon_greedy_instr_selection(self, epsilon=0.1):
        # Check if instruction list is not empty

        selected_items = []
        remaining_items = self.instruction_list.copy()

        for _ in range(self.b):
            if not remaining_items:
                break  # Break if there are no more items to select

            # Exploration: select a random item with probability epsilon
            if random.random() < epsilon:
                random_key = random.choice(list(remaining_items.keys()))
                selected_items.append((random_key, remaining_items.pop(random_key)))
            # Exploitation: select the item with the minimum batch_loss
            else:
                max_key, max_value = max(remaining_items.items(), key=lambda item: item[1]['batch_loss'])
                selected_items.append((max_key, max_value))
                del remaining_items[max_key]
        self.instruction_list = dict(selected_items)

    def optimize(self, iterations=10, batch_size=5, selection_method="greedy"):
        """
        Perform optimization over a set number of iterations to continuously improve
        the generation instructions based on performance analysis.

        :param batch_size:
        :param selection_method:
        :param iterations: Number of optimization cycles to perform.
        """
        selector = self.greedy_instruction_select

        if selection_method == "epsilon_greedy":
            selector = self.epsilon_greedy_instr_selection
        elif selection_method == "ucb":
            selector = self.ucb
        with open('../data/naive_prompts.txt', 'r') as file:
            query_list = [line.strip() for line in file.readlines()]

        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}/{iterations}")

            # Use the current instruction list to generate prompts. If the list is empty, use the initial instruction.
            if not self.instruction_list:
                self.instruction_list[self.initial_instruction] = {}

            # Generate and track prompts using the current set of instructions and external sources
            self.generate_prompts_and_fetch_track(self.instruction_list, random.sample(query_list, batch_size))

            # select instruction in the instruction pool
            if selection_method == "ucb":
                selector(0.8, 1 + iteration * self.b)
            else:
                selector()

            avg_loss = sum([item["batch_loss"] for item in list(self.instruction_list.values())]) / len(
                self.instruction_list)
            print(
                f"Iteration {iteration + 1}/{iterations} --- Average loss in the instruction where source is 'generated': {avg_loss}")
            if iteration == iterations - 1:
                self.final_instruction = \
                    sorted(list(self.instruction_list.items()), key=lambda x: x[1]['batch_loss'])[-1][0]
            # Step 2: Analyze performance and propose new instructions
            self.analyze_and_propose()

            # Step 3: Maintain a dynamic list of instructions for future iterations
            # The maintain_instructions_list is called within analyze_and_propose()

            # calculate the average loss in the prompts_tracking where sourse is "generated"

            self.query_to_prompt = {}

            # Pause between iterations to avoid rate limiting or overloading (if applicable)
            time.sleep(1)

        print("\nOptimization completed.")

    def print_tracking_info(self):
        for pair in self.query_to_prompt.items():
            print(pair)

    def delete_a(self):
        self.generator.delete_assis()
        self.instruction_modifier.delete_assis()
