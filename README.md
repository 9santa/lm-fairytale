
## Data processing test

Train chars: 5000000\
Valid chars: 500000\
Vocab size:  59\
Train ids:   5000000\
Valid ids:   500000\
Train batch x shape: [32, 128]\
Train batch y shape: [32, 128]\
Valid batch x shape: [32, 128]\
Valid batch y shape: [32, 128]

Decoded sample:

once upon a time there was a little boy named ben. ben loved to explore the world around him. he saw many amazing things, like beautiful vases that were on display in a store. one day, ben was walkin

## What's next

A language model tries to assign probability to a sequence of tokens.
If a token sequence is $$x1,x2,x3,…,xT$$:
then the full probability of the whole sequence is decomposed with the probability chain rule as:
$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, \dots, x_{t-1})$$

A real transformer tries to estimate $$P(x_1, x_2, \dots, x_T)$$ using the entire prefix.

A bigram LM makes a big simplification:
$$P(x_t∣x_1,…,x_t−1)≈P(x_t∣x_t−1)$$
meaning that the next token only depends on the immediately previous one.

So the next step is to build a bigram LM as a baseline model.

## BIRGAM LM RESULTS

Device:      CPU
Train chars: 5000000
val chars: 500000
Vocab size:  59
[step 0] train loss = 4.68456, valid loss = 4.68656
[step 200] train loss = 4.38437, valid loss = 4.38549
[step 400] train loss = 4.11346, valid loss = 4.10946
[step 600] train loss = 3.86518, valid loss = 3.86553
[step 800] train loss = 3.64679, valid loss = 3.64494
[step 1000] train loss = 3.44917, valid loss = 3.44879
[step 1200] train loss = 3.28024, valid loss = 3.27501
[step 1400] train loss = 3.13076, valid loss = 3.12728
[step 1600] train loss = 2.99915, valid loss = 2.99553
[step 1800] train loss = 2.88494, valid loss = 2.8799
[step 2000] train loss = 2.78726, valid loss = 2.78673
[step 2200] train loss = 2.70532, valid loss = 2.70173
[step 2400] train loss = 2.63253, valid loss = 2.62917
[step 2600] train loss = 2.56928, valid loss = 2.56714
[step 2800] train loss = 2.51767, valid loss = 2.51829
[step 2999] train loss = 2.47585, valid loss = 2.47685

=== GENERATED SAMPLE ===
once upon a timery. o t54dor542�vely. cowad
ted t5qund. bum g tr. fr thaledr th��s<m! o oy o g he. d.
< plen fan.5noued hew w fr li�cay n'theyom fay tomorulle ppin bungricranell cindey seasoour,>
tjun id ly theswithtito g bidanstin'te. thithandscke, teadsatela lpied�g cidd in cal�4s b9f�8jinng. hai bid ho tofe ndey


## Switch to a Transformer

The bigram model could only learn local transition statistics like:
 - after `q`, `u` is likely
 - after space, a letter is likely
 - after `.`, a newline or space is likely

But it could not learn:
 - what word it is currently inside
 - what happened 4 tokens ago
 - subject-verb agreement
 - whether the story is about Ben or Lucy
 - whether we are inside dialogue
 - whether the sentence already started and needs to end

### What the Transformer is trying to do instead

The Transformer says:
 `At position t, i want a representation that summarizes the relevant parts of the earlier tokens 1, 2, ..., t`
Then from that representation, it predicts the next token.

So instead of

$$[\text{logits}_t = W[x_t]]$$

we get something more like

$$[h_t = f(x_1,\dots,x_t)]$$

$$[\text{logits}_t = W{out} * h_t]$$
where $h_t$ is a learned vector representation of the token at position $t$, with 'embedded' context.





Why GELU?
For Transformers, GELU is a very common activation function. ReLU would also work.







### The full TransformerLM model body
idx [B, T]
  -> token + position embeddings [B, T, d_model]
  -> TransformerBlock 1 [B, T, d_model]
  -> TransformerBlock 2 [B, T, d_model]
  -> final LayerNorm [B, T, d_model]
  -> Linear head [B, T, vocab_size]



Bigram LM:
  3000 steps, valid loss ≈ 2.48

Tiny Transformer:
  200 steps, valid loss ≈ 2.33
  3000 steps, valid loss ≈ 1.35

We can see that the Transformer model already beats Bigram model after just 200 steps.
Comparing at the same number of steps: Transformer's validation loss is almost 2x times lower (45.6% decrease).


Device:      CUDA
Train chars: 5000000
Valid chars: 500000
Vocab size:  59
[step 0] train loss = 4.2617, valid loss = 4.26197
[step 200] train loss = 2.338, valid loss = 2.33022
[step 400] train loss = 2.25016, valid loss = 2.24702
[step 600] train loss = 2.16086, valid loss = 2.15615
[step 800] train loss = 2.0447, valid loss = 2.03025
[step 1000] train loss = 1.89214, valid loss = 1.88649
[step 1200] train loss = 1.7727, valid loss = 1.76056
[step 1400] train loss = 1.6965, valid loss = 1.67166
[step 1600] train loss = 1.61834, valid loss = 1.59667
[step 1800] train loss = 1.56081, valid loss = 1.55039
[step 2000] train loss = 1.50914, valid loss = 1.50602
[step 2200] train loss = 1.47255, valid loss = 1.45351
[step 2400] train loss = 1.42752, valid loss = 1.43019
[step 2600] train loss = 1.41459, valid loss = 1.40535
[step 2800] train loss = 1.38272, valid loss = 1.37057
[step 2999] train loss = 1.34599, valid loss = 1.35413

=== GENERATED SAMPLE ===
once upon a time. they were was a big for hore. the saw a nice. look, "thanked the sleed and the bird the bird the prock the park the heard the was on the dad. he some and said, "he want he was beat the day and stauch out the car beat and the wated the harry. the climped to hell food sun what her saw a bird the toy want to be goome stoppes the drous the friends. they was show would a sun and the close and feeliff


That is a huge improvement. It shows exactly what we wanted to demonstrate:

self-attention + context beats first-order transition statistics.

The generated sample also changed qualitatively. It is still broken, but now it has:

recognizable words
sentence-like rhythm
quoted speech
repeated story-ish phrases
local grammar fragments
some TinyStories style

Examples:

they were
saw a nice
said
friends


char-transformer-2layer-d128-h4-context128-10k
valid loss: 1.0858
=== GENERATED SAMPLE ===
once upon a time, there was a big tree and said, "i will be named and the bird not with me, in a small dog named broke and dad."
the little boy was sad and the cat was scary and the face and the sky. the fun was too saw the tree and wanted to her toys. the saw a big became because he was so happy.
the draw and wanted to do share it and said some of his friends. it was a big cat was angreed the bear. they all beca
