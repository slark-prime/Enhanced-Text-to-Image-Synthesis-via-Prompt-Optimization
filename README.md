# To Run in Notebooks: 

```shell
!git clone https://github.com/slark-prime/MAAI_Optimizer.git
%cd MAAI_Optimizer
!git clone https://github.com/tgxs002/HPSv2.git
%cd HPSv2
!pip install -e .
%cd ..
!pip install hpsv2
!pip install openai
!pip install diffusers

```

# Experiments
- [ ] UCB batch 1 with lexica
- [ ] UCB batch 3 with lexica
- [ ] UCB batch 5 with lexica
- [ ] UCB batch 3 without lexica
- [ ] greedy batch 3 with lexica
- [ ] epsilon greedy batch 3 with lexica
- [ ] gpt3.5 baseline 
