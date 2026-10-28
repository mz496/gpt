Following Karpathy's https://karpathy.ai/zero-to-hero.html -> Let's build GPT



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

Upping the parameters to match Karpathy's:
- After 30 minutes of training on my macbook, I killed it early because it already reached a reasonable value, not sure exactly why my numbers were consistently slightly lower than Karpathy's though. Perhaps I was pushing the parameters harder somehow.
- `step=1500 loss={'train': 1.2820947170257568, 'validation': 1.4961657524108887}`
