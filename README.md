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
- [x] UCB batch 1 with lexica （anthony-done now)
- [x] UCB batch 3 with lexica (partly with 7/10 iterations)
- [x] UCB batch 5 with lexica (anthony-done 4 iters)
- [x] UCB batch 3 without lexica 
- [ ] greedy batch 3 with lexica
- [x] epsilon greedy batch 3 with lexica
- [ ] gpt3.5 baseline 
