from src.pipeline import AgentsPipline


def experiment_batch_sizes():
    batch_sizes = [1, 3, 5]

    score_trend_batch = []
    final_instruct_s = []
    for batchsize in batch_sizes:
        print(f"this is testing the batch_size:{batchsize}")
        pipe = AgentsPipline(instr_list_length=3)
        pipe.optimize(batch_size=batchsize)
        score_trend_batch.append(pipe.training_p_s)
        final_instruct_s.append(pipe.final_instruction)


def experiment_selection_method():
    selection_method = ["ucb", "greedy", "epsilon_greedy"]

    score_trend_select_method = []
    final_instruct_method = []

    for method in selection_method:
        print(f"this is testing the selection method:{method}")
        pipe = AgentsPipline(instr_list_length=3)
        pipe.optimize(selection_method=method)
        score_trend_select_method.append(pipe.training_p_s)
        final_instruct_method.append(pipe.final_instruction)

def gpt3.5_baseline():
    pass


