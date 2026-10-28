Following Karpathy's https://karpathy.ai/zero-to-hero.html -> Let's build GPT

2026-10-23 goal:

Project 1: Character-level GPT (10 hrs)
- [ ] Follow "Let's build GPT" on TinyShakespeare, using nanoGPT as a reference.
- [ ] Change hyperparameters one at a time and note what each did.
Done when: validation loss is roughly 1.5, the model generates Shakespeare-ish text, and you've written one paragraph on what each hyperparameter did.

Project 2: Transformer from scratch, verified against GPT-2 (10 hrs)
- [ ] Do the transformer-from-scratch chapter of ARENA (chapter 1).
- [ ] Implement attention, MLP, layernorm, and embeddings yourself.
- [ ] Load GPT-2 small weights into your implementation.
Done when: your logits match Hugging Face's GPT-2 to a tight tolerance.

Project 3: Supervised finetuning (10 hrs)
- [ ] Finetune Qwen2.5-0.5B on a small instruction dataset using TRL's SFTTrainer, with the HF LLM course as your guide.
- [ ] Compare base and finetuned outputs on 20 prompts.
- [ ] Port to Thinking Machines' Tinker API.
Done when: you have a W&B run, a Hub model, and a short note on where finetuning helped and where it didn't.


# Lab notebook

Toward the end of the video, Karpathy starts speedrunning some things that are not super obvious in the paper (more so general neural net training tricks) so I decided to log each of the changes:

With 5000 steps,

Adding residual pathways:
- train loss 1.91, val loss 2.04

Adding two layer norms around the self-attention and feed forward blocks:
- train loss 1.91, val loss 2.03

Adding a last layer norm around transformer blocks:
- train loss 1.89, validation loss 2.01

Adding dropout with p=0.2:
- train loss 2.016, validation loss 2.136
- I think I need to start scaling up now

