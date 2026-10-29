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

### Oct 28

Implemented the byte-pair encoding tokenizer, and refactored the char-based tokenizer behind a `tokenizer.py` interface.

The result of BPE on the tiny shakespeare data with a target vocab size of 300, this looks reasonable so far:

```
uv run --locked -m src.tiny_shakespeare.tokenizers.tokenize_check 
294896 chars -> 208317 tokens
merges (unrolled to bytes):
  256: 101,32 ('e ')
  257: 116,104 ('th')
  258: 116,32 ('t ')
  259: 115,32 ('s ')
  260: 100,32 ('d ')
  261: 44,32 (', ')
  262: 111,117 ('ou')
  263: 101,114 ('er')
  264: 105,110 ('in')
  265: 121,32 ('y ')
  266: 97,110 ('an')
  267: 58,10 (':\n')
  268: 111,114 ('or')
  269: 101,110 ('en')
  270: 111,32 ('o ')
  271: 10,10 ('\n\n')
  272: 97,114 ('ar')
  273: 32,257 (' th')
  274: 111,110 ('on')
  275: 104,97 ('ha')
  276: 108,108 ('ll')
  277: 44,10 (',\n')
  278: 46,271 ('.\n\n')
  279: 121,262 ('you')
  280: 104,105 ('hi')
  281: 101,115 ('es')
  282: 116,270 ('to ')
  283: 114,32 ('r ')
  284: 110,111 ('no')
  285: 73,32 ('I ')
  286: 111,102 ('of')
  287: 101,97 ('ea')
  288: 266,260 ('and ')
  289: 276,32 ('ll ')
  290: 111,109 ('om')
  291: 264,103 ('ing')
  292: 119,105 ('wi')
  293: 257,256 ('the ')
  294: 32,109 (' m')
  295: 116,105 ('ti')
  296: 83,267 ('S:\n')
  297: 118,256 ('ve ')
  298: 111,119 ('ow')
  299: 115,116 ('st')
```

