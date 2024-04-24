import random
import hpsv2
import os
import time
from openai import OpenAI



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


class Scorer:
    def __init__(self):
        pass

    @staticmethod
    def get_loss(images, origin_prompt):  # generator loss
        result = hpsv2.score(images, origin_prompt, hps_version="v2.0")[0] * 100
        # loss = np.log(result[0]) + np.log(1 - result[1])

        print(f"score:{result}")
        return result

    @staticmethod
    def get_random_loss():
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
        analysis_request += f"\nThis is the generator instruction:{instruction} \n and first corresponding generated low score prompts group:"
        for i, (obj, prompt) in enumerate(lowscore_prompt_batch):
            analysis_request += f"\nlow_score_object{i}:{obj}, low_score_generated_prompt:{prompt[0]}, score:{prompt[1]}"

        analysis_request += f"\nbelow is high score prompts group:"
        for j, (obj, prompt) in enumerate(highscore_prompt_batch):
            analysis_request += f"\nhigh_score_object{j}:{obj}, high_score_prompt:{prompt[0]},score:{prompt[1]}"
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
