import json
import os
import random

from src.agents import Generator, Scorer
from src.pipeline import SD, AgentsPipline


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


API_KEY = ""
ORG = ""


with open('data/naive_prompts.txt', 'r') as file:
    query_list = [line.strip() for line in file.readlines()]

diffuser = SD()
scorer = Scorer()


def gpt_baseline_test(queries):
    print("Running GPT-3 baseline...")

    os.makedirs("results/test_SD_images", exist_ok=True)

    baseline_generator = Generator(api_key=API_KEY, org=ORG)
    score_sum = 0
    image_scores = {}  # Dictionary to track scores for each image

    for i, query in enumerate(queries):
        baseline_instruction = f"This is the original prompt that you need to carefully refine, Prompt or subject to refine : {query}"

        baseline_generator.add_instruction(baseline_instruction)
        generated_prompt = baseline_generator.get_last_run_response()
        print(f"Generated prompt: {generated_prompt}")

        image = diffuser.pipe(generated_prompt).images[0]
        image_path = f"results/test_SD_images/image_{i}.png"
        image.save(image_path)  # Assuming the image object has a save method

        score = scorer.get_loss(image_path, query)  # Assuming get_loss method calculates the loss
        print(f"Score: {score}")

        score_sum += score
        image_scores[image_path] = score

    average_score = score_sum / len(queries)
    print(f"Average score: {average_score}")

    # Save the scores for future analysis
    with open("results/test_SD_images/score_log.txt", "w") as file:
        for path, score in image_scores.items():
            file.write(f"{path}: {score}\n")


def lexica_baseline_test(queries):
    print("Running Lexica baseline...")

    os.makedirs("results/test_lexica_images", exist_ok=True)

    score_sum = 0
    image_scores = {}  # Dictionary to track scores for each image

    for i, query in enumerate(queries):
        # Fetching a prompt based on the query from a JSONL file
        fetched_prompt = random.sample(fetch_prompts_from_jsonl(query, "data/lexica_prompts.jsonl"))
        print(f"Fetched prompt: {fetched_prompt}")

        # Assuming a generate_image method in a Diffuser class that returns an image object
        image = diffuser.pipe(fetched_prompt).images[0]
        image_path = f"results/test_lexica_images/image_{i}.png"
        image.save(image_path)  # Assuming the image object has a save method

        # Assuming get_loss method calculates the loss based on the image and original query
        score = scorer.get_loss(image_path, query)
        print(f"Score: {score}")

        score_sum += score
        image_scores[image_path] = score

    average_score = score_sum / len(queries)
    print(f"Average score: {average_score}")

    # Save the scores for future analysis
    os.makedirs("results/test_lexica_images", exist_ok=True)  # Ensure directory exists
    with open("results/test_lexica_images/score_log.txt", "w") as file:
        for path, score in image_scores.items():
            file.write(f"{path}: {score}\n")


def gpt_ucb_instruction_test(queries, instruction, test_id):
    print(f"Running GPT-3.5 with UCB {test_id} instruction baseline...")
    baseline_generator = Generator(api_key=API_KEY, org=ORG)
    score_sum = 0
    image_scores = {}  # Dictionary to track scores for each image

    os.makedirs(f"results/test_ucb{test_id}_images", exist_ok=True)

    for i, query in enumerate(queries):
        refined_instruction = f"{instruction}, Prompt or subject to refine : {query}"

        baseline_generator.add_instruction(refined_instruction)
        generated_prompt = baseline_generator.get_last_run_response()
        print(f"Generated prompt: {generated_prompt}")

        image = diffuser.pipe(generated_prompt).images[0]
        image_path = f"results/test_ucb{test_id}_images/image_{i}.png"
        image.save(image_path)  # Assuming the image object has a save method

        score = scorer.get_loss(image_path, query)  # Assuming get_loss method calculates the loss
        print(f"Score: {score}")

        score_sum += score
        image_scores[image_path] = score

    average_score = score_sum / len(queries)
    print(f"Average score: {average_score}")

    # Save the scores and images for future analysis
    os.makedirs(f"results/test_ucb{test_id}_images", exist_ok=True)
    with open(f"results/test_ucb{test_id}_images/score_log.txt", "w") as file:
        for path, score in image_scores.items():
            file.write(f"{path}: {score}\n")


def benchmark():
    random.seed(777)
    to_query = random.sample(query_list, 10)

    # gpt_baseline_test(to_query)
    # lexica_baseline_test(to_query)

    ucb_1_instruction = "Integrate exercises that challenge writers to distill their descriptions to the most essential elements while effectively evoking the desired emotions, reinforcing the lesson on brevity and precision in storytelling. Emphasize the use of impactful language and imagery to succinctly capture the essence of a scene and immerse the reader in a cohesive narrative that evokes awe, wonder, and exploration, ultimately igniting feelings of exhilaration and reverence for the boundless beauty and possibilities within the depicted setting"
    ucb_3_instruction = "Structure prompts for a coherent flow of ideas and imagery, ensuring immersive and compelling descriptions with a clear and "
    ucb_5_instruction = ""

    # gpt_ucb_instruction_test(to_query, ucb_1_instruction, 1)
    gpt_ucb_instruction_test(to_query, ucb_3_instruction, 3)
    gpt_ucb_instruction_test(to_query, ucb_5_instruction, 5)


if __name__ == '__main__':
    benchmark()
