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

```
'buck me; being his are did so with ston.

LOUCESTER:
I do him weeth do him.

LARY ANNE:
A worthy souls thinking to brought the killy:
You tribes of like for a stand in thing
Bestay those hath Rome that to our causes.

GLOUCESTER:
Your cersel'st leave smook delow,
Arest in for every thousats.

GLOUCESTER:
Cot of no vallant alone, souf: as you, have but
I befIt it a do. Wors you bear, to bid you,
But to to you, as not in actats those,
When my praicely mon this venigh'd the warling Rome,
What in a widone? whethou heart to the say knot,
Whe woundow'st hundle the beir will joy things;
For voothous sake pobers, the own soown,
Could of Roman, by morrought.s I crave thes senderand
To know the pity and for the strews to loving
Will revenge fullie, with Jone.
I loved me, Marcius, with thou; I nay't thou have
Bad upon eyear trus you.

First Lord:
Let morkingham, I saw, as enterflew-corsen virtict teas,
Against the world ford then masters,
My povoker and give by father; ther'd, so fe
Whose inpect
```
