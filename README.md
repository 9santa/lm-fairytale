
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
