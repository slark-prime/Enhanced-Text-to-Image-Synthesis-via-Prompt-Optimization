import json
import random
import re
import requests
import HPSv2.hpsv2 as hpsv2
import os
import time
from openai import OpenAI
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
import torch

NEW_INSTRUCTION_NUM = 5


class Agent:
    def __init__(self, api_key=None, org=None) -> None:
        self.client = OpenAI(api_key=api_key, organization=org)
        self.agent_type = None
        self.thread = None
        self.assistant = None
        self.near_run = None
        self.initial_sys_prompt = None

    def _create_assist(self):
        pass

    def creat_assisstance(self):
        self._create_assist()
        self.thread = self.client.beta.threads.create()
        print(self.assistant.id)

    def add_instruction(self, messg):
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=messg
        )
        self.near_run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,

        )

    def get_last_run_response(self):
        run = self.client.beta.threads.runs.retrieve(
            thread_id=self.thread.id,
            run_id=self.near_run.id
        )

        while run.status != "completed":
            if run.status == "failed":
                print(run.status)
                break
            if run.status == "cancelled":
                print("the run is cancelled")
                break
            if run.status == "expired":
                print("the run is expired")
                break
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.near_run.id
            )
        print(run.status)
        if run.status == "completed":
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id
            )
            # print(messages)
            # print(f"{self.agent_type} output:{messages.data[0].content[0].text.value}")
        return messages.data[0].content[0].text.value

    def delete_assis(self):
        self.client.beta.assistants.delete(self.assistant.id)


class Generator(Agent):
    def __init__(self, api_key=None, org=None) -> None:
        super().__init__(api_key, org)
        self.agent_type = "Generator"
        self.initial_sys_prompt = '''you are a prompt generator, whose role is to take a basic, brief snippet of text, a word, a sentence or two and skillfully enhance it into a refined, detailed prompt. This enriched prompt,
          while preserving the original text's semantic essence, is designed for creating images with a text to image model.
              '''
        variable_value = os.environ.get('G_assistance_id')
        print(f"env var:{variable_value}")

        if variable_value is None:
            self.creat_assisstance()
        else:
            self.assistant = self.client.beta.assistants.retrieve(variable_value)
            self.thread = self.client.beta.threads.create()

    def _create_assist(self):
        self.assistant = self.client.beta.assistants.create(
            name=self.agent_type,
            instructions=self.initial_sys_prompt,
            model="gpt-3.5-turbo-0125",
            tools=[{"type": "code_interpreter"}]
        )


class Discriminator:
    def __init__(self):
        pass

    def get_loss(self, images, origin_prompt):  # generator loss
        result = hpsv2.score(images, origin_prompt, hps_version="v2.0")[0] * 100
        # loss = np.log(result[0]) + np.log(1 - result[1])

        print(f"score:{result}")
        return result

    def get_random_loss(self):
        return random.uniform(0, 1)


class InstructionModifier(Agent):
    def __init__(self, api_key=None, org=None) -> None:
        super().__init__(api_key, org)
        self.agent_type = "instruction_modifier"
        self.creat_assisstance()

    def _create_assist(self):
        self.assistant = self.client.beta.assistants.create(
            name=self.agent_type,
            instructions='''Given your role as a instruction modifier with the task of altering instruction for a generator which is a LLM whose role is to write good prompt for image generation,
        while ensuring diversity, variation, more precise and accurate description. Your generated instruction should be as short as possible, mdofied based on the performance analysis. Your output should only contains the new general instruction that can be applied to every simple prompts or subjects
              ''',
            model="gpt-3.5-turbo-0125",
        )


class GradientCalculator(Agent):
    def __init__(self, api_key=None, org=None) -> None:
        super().__init__(api_key, org)
        self.agent_type = "gradient_calculator"
        variable_value = os.environ.get('im_assistance_id')
        print(f"env var:{variable_value}")

        if variable_value is None:
            self.creat_assisstance()
        else:
            self.assistant = self.client.beta.assistants.retrieve(variable_value)

    def _create_assist(self):
        self.assistant = self.client.beta.assistants.create(
            name=self.agent_type,
            instructions="Analyze the provided low score promts batch generated by a specific instruction and high score prompts batch to infer reasons for the performance for low score group. Generate suggestions for the improvement of the instruction. Remember you are trying to infer a general weakness behind instead of the subject specific weakness, so better prevent involving the subject ",
            model="gpt-3.5-turbo-0125",
        )

    def analyze_prompts(self, instruction, lowscore_prompt_batch, highscore_prompt_batch):
        if not self.thread:
            self.create_assistant()

        # Compile all prompts and scores into a single analysis request
        analysis_request = """Analyze the following low score and high score batch, each prompt with corresponding scores. And infer what's wrong with the instruction generating low score batch prompt to suggest the improvement of the instruction :
      For your answer use the format:\nInference 1: your inference \nInferecne 2: your inference\n Inference n: your inference...
      \nImprovement 1: you suggested improvement correspond to inference 1  \nImprovement 2: you suggested improvement correspond to Inference 2\n Improvement n: you suggested improvement correspond to inference n..."""
        analysis_request += "\n".join([
                                          f"This is the generator instruction:{instruction} \n and first corresponding generated low score prompts group:"])
        for i, (obj, prompt) in enumerate(lowscore_prompt_batch):
            analysis_request += "\n".join(
                [f"low_score_object{i}:{obj}, low_score_generated_prompt:{prompt[0]}, score:{prompt[1]}"])

        analysis_request += "\n".join([f"below is high score prompts group:"])
        for j, (obj, prompt) in enumerate(highscore_prompt_batch):
            analysis_request += "\n".join(
                [f"high_score_object{j}:{obj}, high_score_prompt:{prompt[0]},score:{prompt[1]}"])
        print(analysis_request)

        # print(f"Analysis Request: {analysis_request}")
        # Send the compiled request to the LLM
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=analysis_request
        )
        self.near_run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
        )
        return self.get_last_run_response()


import heapq


class SD:
    def __init__(self) -> None:
        self.model_id = "stabilityai/sdxl-turbo"
        self.pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, torch_dtype=torch.float16,
                                                              variant="fp16").to("cuda")


class AgentsPipline:
    def __init__(self) -> None:
        self.generator = Generator(API_KEY, ORG)
        self.discriminator = Discriminator()
        self.instruction_modifier = InstructionModifier(API_KEY, ORG)
        self.Diffusion_M = SD()
        self.query_to_prompt = {}
        self.instruction_list = {}
        self.gradient_calculator = GradientCalculator(API_KEY, ORG)
        self.b = 5  # instruction_list length
        self.initial_instruction = "This is the original prompt that you need to carefully refine"

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
        # TODO: try different system prompt or instruction to force certain format of output,
        # TODO: apply regex to extract it.
        return generated_prompt

    def IM_output(self, meg):
        self.instruction_modifier.add_instruction(meg)
        new_instruction = self.instruction_modifier.get_last_run_response()
        # print(f"IM_output: New instruction: {new_instruction}")
        return new_instruction

    def loss_compute_and_track_batch(self, prompt_info):
        """
        Computes the loss for a batch of prompts and tracks the results.

        :param prompts_batch: A list of dictionaries, each containing prompt, instruction, and source information.
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
        loss = self.discriminator.get_loss(image_path, prompt)

        # batch_losses.append(loss)

        # self.prompts_tracking.append({
        # 'prompt': prompt,
        # 'instruction': instruction if source == 'generated' else None,
        # 'loss': loss,
        # 'source': source
        # })
        # Calculate the average loss for the batch
        # batch_average_loss = sum(batch_losses) / len(batch_losses)
        print(f" loss: {loss}")

        return loss

    # def loss_compute_and_track(self, prompt, instruction, query, source='generated'):
    #     # This is for single prompt loss calculation
    #     image = self.Diffusion_M.pipe(prompt).images[0]
    #     file_name = f"{query}_{source}_{instruction[:10]}.png" if source == 'generated' and instruction is not None \
    #         else f"{query}_{source}_lexica_{prompt[:5]}.png"
    #     os.makedirs("./SD_images", exist_ok=True)
    #     image_path = os.path.join("./SD_images", file_name)
    #     image.save(image_path)
    #     loss = self.discriminator.get_loss(image_path, prompt)

    #     print(f"source: {source}")

    #     # loss = self.discriminator.get_random_loss()

    #     # loss = 0 if source == "generated" else loss

    #     self.prompts_tracking.append({
    #         'prompt': prompt,
    #         'instruction': instruction if source == 'generated' else None,
    #         'loss': loss,
    #         'source': source
    #     })

    def find_extreme_instruction(self, n=1):
        # """

        # Finds the tracks for the n highest scoring prompts and n lowest scoring instructions.

        # :param n: Number of top and bottom scores to retrieve.
        # :return: A tuple containing lists of the n highest scores and n lowest scores.
        # """

        '''
        Locate n lowest batch loss instruction
        '''
        if len(self.instruction_list) == 0 or n < 1:
            print("Invalid input or no prompts have been tracked yet.")
            return [], []

        # Sort the tracked prompts by loss
        sorted_instruction = sorted(list(self.instruction_list.items()), key=lambda x: x[1]['batch_loss'])

        # Retrieve the n highest scores (lowest losses) and n lowest scores (highest losses)

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
                inst_batch_loss += loss_q
                prompt_score_tuple = (generated_prompt, loss_q)
                generate_prompts = instruction_dic.get("querys")
                generate_prompts[q] = prompt_score_tuple
                print(prompt_score_tuple)
                # add into all prompt pool

                if q not in self.query_to_prompt:
                    self.query_to_prompt[q] = []
                self.query_to_prompt[q].append(prompt_score_tuple)

                # fetch prompt from lexico and add the tuple into all prompt pool
                lexica_prompts = self.fetch_prompts_from_lexica(q)
                lexica_prompts = random.sample(lexica_prompts, k=min(5, len(lexica_prompts)))
                for lex_prompt in lexica_prompts:
                    lexprompt_score_tuple = (lex_prompt, self.loss_compute_and_track_batch(
                        {'prompt': lex_prompt, 'instruction': None, 'source': 'lexica', "query": q}))
                    self.query_to_prompt[q].append(lexprompt_score_tuple)
                self.query_to_prompt[q] = sorted(self.query_to_prompt[q], key=lambda x: x[1])

            instruction_dic["batch_loss"] = inst_batch_loss / len(query)
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
            for query, (gener_prompt, loss) in lowest_inst_query_prompts_pair:
                highest_score_prompt_to_query = self.query_to_prompt[query][-1]
                if gener_prompt[0] != highest_score_prompt_to_query[0]:
                    highest_score_prompt.append((query, highest_score_prompt_to_query))
                    lowest_score_prompt_for_instr.append((query, gener_prompt))
            reasons_summary = self.gradient_calculator.analyze_prompts(lowest_instruction[0],
                                                                       lowest_score_prompt_for_instr,
                                                                       highest_score_prompt)

            print(f"analyze_and_propose: Summary of Reasons for Scores:{reasons_summary}")

            improvements = self.extract_improvement(reasons_summary)

            print(f"improvement list: {improvements}")
            for improvement in improvements:
                print("analyze_and_propose: improvement:", improvement)
                # Assuming 'instruction' is available in the prompt_info structure
                original_instruction = lowest_instruction

                # Preparing feedback for instruction modification
                analysis_feedback = f"Based on improvemnt suggestion: {improvement}, consider improving the instruction: '{original_instruction}' , while try to avoid mention specific information about the prompt'"
                new_instruction = self.IM_output(analysis_feedback)
                print("analyze_and_propose: Updated Instruction:", new_instruction)
                self.instruction_list[new_instruction] = {"batch_loss": 0, "querys": {}}

            del self.instruction_list[lowest_instruction[0]]

    def maintain_instructions_list(self):
        # TODO: this is
        # Ensure the list is capped at self.b elements
        if len(self.instruction_list) > self.b:
            sampled_items = random.sample(list(self.instruction_list.items()), self.b)
            self.instruction_list = dict(sampled_items)

    def greedy_instruction_select(self):
        top_n_inst = heapq.nlargest(min(self.b, len(self.instruction_list)), self.instruction_list.items(),
                                    key=lambda item: item[1]['batch_loss'])
        self.instruction_list = dict(top_n_inst)

        print(f"top_instr:{self.instruction_list}")

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

    def optimize(self, iterations=10, batch_size=3):
        """
        Perform optimization over a set number of iterations to continuously improve
        the generation instructions based on performance analysis.

        :param iterations: Number of optimization cycles to perform.
        :param query: Query to fetch external prompts for additional variety.
        """

        with open('naive_prompts.txt', 'r') as file:
            query_list = [line.strip() for line in file.readlines()]

        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}/{iterations}")

            # Use the current instruction list to generate prompts. If the list is empty, use the initial instruction.
            if not self.instruction_list:
                self.instruction_list[self.initial_instruction] = {}

            # Generate and track prompts using the current set of instructions and external sources
            self.generate_prompts_and_fetch_track(self.instruction_list, random.sample(query_list, batch_size))

            # select instruction in the instruction pool
            self.greedy_instruction_select()

            # Step 2: Analyze performance and propose new instructions
            self.analyze_and_propose()

            # Step 3: Maintain a dynamic list of instructions for future iterations
            # The maintain_instructions_list is called within analyze_and_propose()

            # calculate the average loss in the prompts_tracking where sourse is "generated"
            avg_loss = sum([item["batch_loss"] for item in list(self.instruction_list.values())]) / len(
                self.instruction_list)
            print(
                f"Iteration {iteration + 1}/{iterations} --- Average loss in the instruction where source is 'generated': {avg_loss}")
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


if __name__ == '__main__':
    pipe = AgentsPipline()
    pipe.optimize()
