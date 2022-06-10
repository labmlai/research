# Rotary Positional Embeddings with Relative distances (RoPER)

Rotary Positional Embeddings (RoPE) 
([Su et al., 2021](https://papers.labml.ai/paper/a0708356b79611ebbd9b8f626bc6f333))
use relative positions in attention score calculation.
However, in RoPE, no positional information (relative or otherwise) is added to embeddings. 

We investigate adding relative positions of the tokens it pays attention to, to the values of the attention calculation.
We call this method RoPER. We evaluate the method and compare it with RoPE in a couple of algorithmic
tasks and show that RoPER performs better.
We also try it on a language model and show that RoPER doesn’t degrade performance compared to RoPE.

## Background

RoPE rotates pairs of features in a 2D plane, by an angle that is a multiple of their position in the modeled sequence.
When such a pair participates in dot-product attention only the relative positional difference between
that pair and the other pair of embeddings affects the result. 

RoPE has shown to be successful in language modeling and is widely being used
([Chowdhery et al., 2022](https://papers.labml.ai/paper/eb71aa3ab55611ecac827bce58715ee7)).

It has also been shown that auto-regressive transformers can implicitly acquire positional information 
([Haviv et al., 2022](https://papers.labml.ai/paper/2c364684b15b11ecac827bce58715ee7)).

## Method

RoPER, in addition to using relative positions in the attention score calculation,
adds relative positional information explicitly to value embeddings.
Specifically, it incorporates the relative positions of the tokens paid attention to.

RoPER rotates the values in attention in a similar way to how RoPE rotates keys and queries.
Then, after taking the weighted sum of the values, RoPER rotates the final result in the opposite direction,
which leaves the relative positional differences between the current query sequence
position and sequence positions of the values, in the final result of the attention.

### Relative distances in embeddings

For any head, let $a_{n,i}$ be the attention from query position $n$ to value position $i$,
and $v_i$ be the value embeddings at position $i$. Let's denote individual features
as $v^{(1)}_i, v^{(2)}_i, \dots$.

Normally, we would take the weight sum of value embeddings
$$o^{(j)}_n = \sum_i a_{n,i} v^{(j)}_i$$

This doesn't explicitly add any distance information about the positions $i$ to final
result $o^{(j)}_n$.
RoPER pairs features like RoPE and transform them.
For a pair $v^{(1)}_m$ and $v^{(2)}_m$ it transforms them by
 $RoPE\big(v^{(1)}_m, v^{(2)}_m, m\big)$.

Let us donate the transformed features with $\hat{v}^{(1)}_m, \hat{v}^{(2)}_m$.
Then it rotates the weighted sum $\hat{o}^{(j)}_n$ in the the reverse direction with
 $RoPE\big(\hat{o}^{(1)}_n, \hat{o}^{(2)}_n, -n\big)$.
*Note the *$-n$.

Note that,

$$
\begin{align}
RoPE\big(x^{(1)}_m, x^{(2)}_m, m\big) &=
\begin{pmatrix}
\cos m \theta & - \sin m \theta \\
\sin m \theta & \cos m \theta
\end{pmatrix}
\begin{pmatrix}
x^{(1)}_m \\
x^{(2)}_m \\
\end{pmatrix} \\
&=
\begin{pmatrix}
x^{(1)}_m \cos m\theta - x^{(2)}_m \sin m \theta \\
x^{(2)}_m \cos m\theta + x^{(1)}_m \sin m \theta \\
\end{pmatrix} \\
\end{align}
$$

Final output after with the transformations is,

$$
\begin{align}
RoPE\big(\hat{o}^{(1)}_n, \hat{o}^{(2)}_n, -n\big) &= \\
\begin{pmatrix}
\hat{o}^{(1)}_n \cos n\theta + \hat{o}^{(2)}_n \sin n \theta \\
\hat{o}^{(2)}_n \cos n\theta - \hat{o}^{(1)}_n \sin n \theta \\
\end{pmatrix} \\
\end{align}
$$

*Note that *$\sin (-n \theta) = -\sin n \theta$.
Let's expand the first term $\hat{o}^{(1)}_n \cos n\theta + \hat{o}^{(2)}_n \sin n \theta$,

$$
\begin{align}
\hat{o}^{(1)}_n \cos n\theta + \hat{o}^{(2)}_n \sin n \theta &= \\
\sum_i a_{n,i} \hat{v}^{(1)}_i \cos n\theta + \sum_i a_{n,i} \hat{v}^{(2)}_i \sin n \theta &= \\
\sum_i a_{n,i} \Big( v^{(1)}_i \cos i\theta - v^{(2)}_i \sin i \theta \Big) \cos n\theta &+ \\
\sum_i a_{n,i} \Big( v^{(2)}_i \cos i\theta + v^{(1)}_i \sin i \theta \Big) \sin m \theta &= \\
\sum_i a_{n,i} v^{(1)}_i \Big( \cos i\theta \cos n\theta + \sin i \theta \sin n \theta \Big) &+ \\
\sum_i a_{n,i} v^{(2)}_i \Big( \cos i\theta \sin n\theta - \sin i \theta \cos n \theta \Big) &= \\
\sum_i a_{n,i} v^{(1)}_i \cos (i - n) \theta - \sum_i a_{n,i} v^{(2)}_i \sin (i - n) \theta &= \\
\sum_i a_{n,i} v^{(1)}_i \cos (i - n) \theta - \sum_i a_{n,i} v^{(2)}_i \sin (i - n) \theta
\end{align}
$$

Simiarly we can show the second term is equal to,

$$\sum_i a_{n,i} v^{(1)}_i \cos (i - n) \theta + \sum_i a_{n,i} v^{(2)}_i \sin (i - n) \theta$$

Which gives,

$$
\begin{align}
RoPE\big(\hat{o}^{(1)}_n, \hat{o}^{(2)}_n, -n\big) &= \\
\begin{pmatrix}
\sum_i a_{n,i} v^{(1)}_i \cos (i - n) \theta - \sum_i a_{n,i} v^{(2)}_i \sin (i - n) \theta \\
\sum_i a_{n,i} v^{(1)}_i \cos (i - n) \theta + \sum_i a_{n,i} v^{(2)}_i \sin (i - n) \theta \\
\end{pmatrix} &= \\
\sum_i a_{n,i} RoPE \big (v^{(1)}_i, v^{(1)}_i, (i - n) \theta \big)
\end{align}
$$

That is, the weighted average of values rotated relative to current position.

## Analysis

We compare RoPER with RoPE on a couple of algorithmic tasks. We also show that RoPER performs similar to RoPE on language modeling with a small 200M parameter transformer.

For the Arithmetic Addition and the Substring by Index tasks we use a ~20M parameter model (512 embedding, 6 layers, 8 heads, post layer norm). We sequence of length 641 for training and a batch size of 32.. For both these tasks we report the accuracy of solving the problem after 5,000 training steps. We test the number of correct solutions with random sampling for 128 problems.

For the Substring by Search task we use a ~0.6M parameter model (128 embedding, 3 layers, 4 heads, pre-layer norm). We train of sequences of length 513 with a batch size of 16. and we report the final loss after 65,000 steps.

| Task              | RoPE  | RoPER    |
|-------------------|-------|----------|
|Arithmetic Addition|124.33 |**126.33**|
|Substring by Index |62.00  |**96.11** |
|Substring by Prefix|0.3329 |**0.3205**|

### Arithmetic Addition

In this task we test the model’s capacity to solve step-by-step arithmetic addition.

Problem format:

```
?d=[number1]+[number2]; [steps] d==[final result]#
```

We concatenate multiple problems in the same sequence. We use up to 8 digit numbers for this problem.


Here’s an example:

```
?d=77+38446365; 7e0+5e0+0e0==12e0 and 7e1+6e1+1e1==14e1 and 0e2+3e2+1e2==4e2 and 0e3+6e3+0e3==6e3 and 0e4+4e4+0e4==4e4 and 0e5+4e5+0e5==4e5 and 0e6+8e6+0e6==8e6 and 0e7+3e7+0e7==3e7 and d==38446442#?d=66623+401; 3e0+1e0+0e0==4e0 and 2e1+0e1+0e1==2e1 and 6e2+4e2+0e2==10e2 and 6e3+0e3+1e3==7e3 and 6e4+0e4+0e4==6e4 and d==67024#?d=25481082+3301219; 2e0+9e0+0e0==11e0 and 8e1+1e1+1e1==10e1 and 0e2+2e2+1e2==3e2 and 1e3+1e3+0e3==2e3 and 8e4+0e4+0e4==8e4 and 4e5+3e5+0e5==7e5 and 5e6+3e6+0e6==8e6 and 2e7+0e7+0e7==2e7 and d==28782301#?d=577790+20083146; 0e0+6e0+0e0==6e0 and 9e1+4e1+0e1==13e1 and 7e2+1e2+1e2==9e2 and 7e3+3e3+0e3==10e3 and 7e4+8e4+1e4==16e4 and 5e5+0e5+1e5==6e5 and 0e6+0e6+0e6==0e6 and 0e7+2e7+0e7==2e7 and d==20660936#
```

### Substring by index

In this problem the model has to retrieve a suffix of a string.

Problem Format:
```
?s=’[string]’; s[[index]:]==’[suffix]’#
```

We concatenate multiple problems in the same sequence. We use upto 8 digits number for this problem.

Example:

```
?s='dyjeofuxvejmg'; s[8:]=='vejmg'#?s='syoktpufifxes'; s[0:]=='syoktpufifxes'#?s='xpbssshucfolp'; s[0:]=='xpbssshucfolp'#?s='wmyapfpvbqdih'; s[9:]=='qdih'#?s='pwndggwvwcueg'; s[3:]=='dggwvwcueg'#?s='liphezyxyaigw'; s[10:]=='igw'#?s='nsadijatrvnsg'; s[2:]=='adijatrvnsg'#?s='gbakpiyhgcycd'; s[12:]=='d'#?s='irrpifewmjbpu'; s[7:]=='wmjbpu'#?s='trdpewjdofkbf'; s[0:]=='trdpewjdofkbf'#?s='bwangjzypgdtw'; s[10:]=='dtw'#?s='lmmufvskwuiti'; s[4:]=='fvskwuiti'#?s='opllhzhjmliiu'; s[5:]=='zhjmliiu'#?s='gcakbhiepmknz'; s[0:]=='gcakbhiepmknz'#?s='cjnmugcjhvvla'; s[0:]=='cjnmugcjhvvla'#?s='ternjenddpljw'; s[5:]=='enddpljw'#?s='pvceminiczzjv'; s[10:]=='zjv'#
```

### Substring by Prefix

In this problem the model has to retrieve (complete) fixed length substrings by searching them. 

Problem Format:

```
[random_string]>[substring][random_string]>[substring][random_string]...
```

We randomly select a substring of the current sequence (except `>` tokens), and append it to the sequence, followed by a random string string. This is repeated until the total length is equal to the sequence length.

To make the task harder we only use a small vocabulary of size 4.

### Language modeling

## Discussion
