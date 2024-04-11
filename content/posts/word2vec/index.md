+++
title = 'Word2vec'
date = 2024-04-11T22:54:10+05:30
draft = false
math = true
ShowToc= true
+++
# Word Embeddings
Word embeddings is the idea of representing word in the form of vectors to have some notion of **similarity** and **difference** between them. Say in a finite dimensional vector space, we want the vectors representing "cat" and "feline" to be *closer* together than the vectors representing "human" and "dog". We want to embed some sort of semantic meaning to the numbers consisting of these vectors. 
# One-Hot Encodings
Our first thought would be to just represent each word as a one-hot vector. Representing each word as $\mathbb{R}^{|V| \times 1}$ vector where $V$ is our vocabulary set.
$$
\mathbf{w}^{at}=\begin{bmatrix} 
1\\
0\\
0\\
\vdots \\
0\\
\end{bmatrix}
\space,
\space
\mathbf{w}^{zebra}=\begin{bmatrix} 
0\\
0\\
1\\
\vdots \\
0\\
\end{bmatrix},
\space
...
\space
\mathbf{w}^{tiger}=\begin{bmatrix} 
0\\
0\\
0\\
\vdots \\
1\\
\end{bmatrix}
$$
This word representation does not give any notion of similarity, like:
$$(\mathbf{w}^{hotel})^\intercal \mathbf{w}^{motel}=(\mathbf{w}^{hotel})^\intercal \mathbf{w}^{cat}=0$$
# Goal
Our goal is to maximise the following probability:
$$P(w_1,w_2,...,w_n)=\prod_{i=2}^{n}P(w_i|w_{i-1})$$
where $w_i$ is the $i$'th word of the vocabulary. Now with this framework in mind let's look into some models that can help us with this objective.
# Continuous Bag of Words Model (CBOW)
It involves predicting the centre word from the surrounding context. For each word, we want to learn 2 vectors:
- $\mathbf{v}$ : (input vector) when the word is in context.
- $\mathbf{u}$ : (output vector) when the word in the centre.
For example, taking a sequence:
$$\textnormal{The cat jumped over the puddle}$$
We can treat {"The","cat","over","the","puddle"} as the surrounding words and "jumped" is centre word. 
## Notation
- $\mathbf{w}_i$ : Word $i$ from vocabulary $V$.
- $n$ : Dimension of each word vector
- $\mathcal{V}\in\mathbb{R}^{n\times |V|}$ : Input word matrix.
- $\mathbf{v}_i$ : $i$-th column of $\mathcal{V}$, the input representation of word $w_i$.
- $\mathcal{U}\in\mathbb{R}^{|V|\times n}$ :  Output word matrix
- $\mathbf{u}_i$ : $i$-the row of $\mathcal{U}$, the output word representation of word $w_i$.
## Model
- We generate our one hot representation for the input context size $m$ : $(\mathbf{x}^{(c-m)},...,\mathbf{x}^{(c-1)},\mathbf{x}^{(c+1)},...,\mathbf{x}^{(c-m)}) \in \mathbb{R}^{|V|}$.
- $\mathbf{x}^{(c)}$ is the **centre** word itself.
- Fetch **embedding vectors** for our context words:
$$(\mathbf{v}_{c-m}=\mathcal{V}\cdot \mathbf{x}^{(c-m)},...,\mathbf{v}_{c-1}=\mathcal{V}\cdot \mathbf{x}^{(c-1)},\mathbf{v}_{c+1}=\mathcal{V}\cdot \mathbf{x}^{(c+1)},...,\mathbf{v}_{c+m}=\mathcal{V}\cdot \mathbf{x}^{(c-m)}) \in \mathbb{R}^{n}$$
- Average these vectors to get $\hat{v}$ :
$$\mathbf{\hat{v}}=\frac{\mathbf{v}_{c-m}+...+\mathbf{v}_{c-1}+\mathbf{v}_{c+1}+...+\mathbf{v}_{c+m}}{2m}\in \mathbb{R}^{n}$$
- Generate score vector $\mathbf{z}$ :
$$\mathbf{z}=\mathcal{U}\cdot\mathbf{\hat{v}}\in\mathbb{R}^{|V|}$$
- Turn score into probabilities:
$$\hat{\mathbf{y}}=\textnormal{softmax}(\mathbf{z})\in\mathbb{R}^{|V|}$$
- We want $\mathbf{\hat{y}}$ to match $\mathbf{y}$ which is $\mathbf{x}^{(c)}$ itself.
- We do this through the cross entropy loss:
$$\mathcal{L}(\mathbf{\hat{y}},\mathbf{y})=-\sum_{j=1}^{|V|}y_j\log \hat{y}_j$$
- Since $\mathbf{y}$ is a one hot vector, all the other indexes except the index of the centre word can be ignored in the summation,
$$\mathcal{L}(\mathbf{\hat{y}},\mathbf{y})=-y_i\log\hat{y}_i$$
   where $i$ is the index of the centre word.
- Stochastic gradient descent is used to update matrices $\mathcal{V}$ and $\mathcal{U}$ which make up our word embeddings.
## Optimisation Objective
We want to minimise:
$$J=-\log P(\mathbf{w}_c|\mathbf{w}_{c-m},...,\mathbf{w}_{c-1},\mathbf{w}_{c+1},...,\mathbf{w}_{c+m})$$
We can simplify this log probability to:
$$J = -\log P(\mathbf{u}_c|\mathbf{\hat{v}})$$
where $\mathbf{u}_c$ is the embedding vector of our centre word from our output matrix $\mathcal{U}$.

According to our model, we take the dot product between our embedding of $\mathbf{w}_c$ and $\mathbf{\hat{v}}$ (average of our embedded word vectors) to compute the similarity, from which we can calculate the probability of the centre word occurring given the context words using a softmax score. 

As discussed before, we will just have to consider the index of the centre word to get our loss:
$$-\log \frac{\exp(\mathbf{u}_c^\intercal\mathbf{\hat{v}})}{\sum_{j=1}^{|V|}\exp(\mathbf{u}_j^\intercal\mathbf{\hat{v}})}$$
which can yield us our final function:
$$J =-\mathbf{u}_c^\intercal\mathbf{\hat{v}}+\log\sum_{j=1}^{|V|}\exp(\mathbf{u}_j^\intercal\mathbf{\hat{v}})$$
# Skip-Gram Model
This is the opposite of what CBOW does. It involves given a centre word, we want to predict the surrounding words. The notation for this model is very same as CBOW but our model is slightly different.
## Notation
- $\mathbf{w}_i$ : Word $i$ from vocabulary $V$.
- $n$ : Dimension of each word vector
- $\mathcal{V}\in\mathbb{R}^{n\times |V|}$ : Input word matrix.
- $\mathbf{v}_i$ : $i$-th column of $\mathcal{V}$, the input representation of word $w_i$.
- $\mathcal{U}\in\mathbb{R}^{|V|\times n}$ :  Output word matrix
- $\mathbf{u}_i$ : $i$-the row of $\mathcal{U}$, the output word representation of word $w_i$.
## Model
- We get our one-hot input vector of the centre:
$$\mathbf{x}_c\in\mathbb{R}^{|V|}$$
- We generate our embedded vector from the centre word:
$$\mathbf{v}_c=\mathcal{V}\cdot\mathbf{x}_c\in\mathbb{R}^n$$
- Generate score vector:
$$\mathbf{z}=\mathcal{U}\mathbf{v}_c$$
 - Get probabilities:
$$\mathbf{\hat{y}}=\textnormal{softmax}(\mathbf{z})$$
 - We take this probability distribution and divide it among $2m$ context words:
$$(\mathbf{\hat{y}}^{(c-m)},...,\mathbf{\hat{y}}^{(c-1)},\mathbf{\hat{y}}^{(c+1)},...,\mathbf{\hat{y}}^{(c-m)}) \in \mathbb{R}^{|V|}$$
   These are just the same probability distributions calculated using softmax just repeated $2m$ times.
- We want our probabilities match our true probabilities:
$$(\mathbf{y}^{(c-m)},...,\mathbf{y}^{(c-1)},\mathbf{y}^{(c+1)},...,\mathbf{y}^{(c-m)}) \in \mathbb{R}^{|V|}$$
   which are one hot vectors of the context words.
- Similar to CBOW, we cross-entropy to update our embedding matrices.
## Optimisation Objective
We invoke a **Naive Bayes** assumption to get our probabilities which in this context means given the centre word all output words are completely independent.

So our optimisation objective is to minimise:
$$J = -\log P(\mathbf{w}^{(c-m)},...,\mathbf{w}^{(c-1)},\mathbf{w}^{(c+1)},...,\mathbf{w}^{(c-m)}|\mathbf{w}_c)$$
Since we are taking the Naive Bayes assumption we can write this as:
$$-\log\prod_{j=0,j\neq m}^{2m}P(\mathbf{w}_{c-m+j}|\mathbf{w}_c)$$
Now taking our embedding vectors from the matrices $\mathcal{V}$ and $\mathcal{U}$.
$$-\log\prod_{j=0,j\neq m}^{2m}P(\mathbf{u}_{c-m+j}|\mathbf{v}_c)$$
Taking dot product and softmax:
$$-\log\prod_{j=0,j\neq m}^{2m}\frac{\exp(\mathbf{u}_{c-m+j}^\intercal\mathbf{v}_c)}{\sum_{k=1}^{|V|}\exp(\mathbf{u}_k^\intercal\mathbf{v}_c)}$$
Our final optimisation objective becomes:
$$J=-\sum_{j=0,j\neq m}^{2m}\mathbf{u}_{c-m+j}^\intercal\mathbf{v}_c+2m\cdot\log\sum_{k=1}^{|V|}\exp(\mathbf{u}_k^\intercal\mathbf{v}_c)$$
# Negative Sampling
In the objective function of the previous two models, the summation over $|V|$ is huge and is computationally expensive when considering our vocabulary size can be in the millions. A simple idea is to approximate it.

Instead of summing over the entire vocabulary, we can sample various negative examples.
## Notation
- $\mathbf{w}_i$ : Word $i$ from vocabulary $V$.
- $n$ : Dimension of each word vector
- $\mathcal{V}\in\mathbb{R}^{n\times |V|}$ : Input word matrix.
- $\mathbf{v}_i$ : $i$-th column of $\mathcal{V}$, the input representation of word $w_i$.
- $\mathcal{U}\in\mathbb{R}^{|V|\times n}$ :  Output word matrix
- $\mathbf{u}_i$ : $i$-the row of $\mathcal{U}$, the output word representation of word $\mathbf{w}_i$.
- $\mathbf{c}$ : Context word.
- $\theta$ : Parameters of our model, $\mathcal{V}$ and $\mathcal{U}$.
- $\mathcal{D}$ : Training corpus.
- $\hat{\mathcal{D}}$ : Not training corpus. Will have unnatural sentences which should not exist like, "boil water hell."
## Model
Consider a word pair $(\mathbf{w},\mathbf{c})$. Did this pair from the training corpus? Let probability it came from the corpus ($\mathcal{D}$) be
$$P(X=1|\mathbf{w},\mathbf{c};\theta)$$
and that it came from "not" the training corpus be:
$$P(X=0|\mathbf{w},\mathbf{c};\theta)$$
We can easily model $P(X=1|\mathbf{w},\mathbf{c};\theta)$ as a sigmoid.
$$P(X=1|\mathbf{w},\mathbf{c};\theta)=\sigma(\mathbf{v}_c^\intercal\mathbf{v}_w)$$
Our goal:
- Maximise $P(X=1|\mathbf{w},\mathbf{c};\theta)$ if $\mathbf{w}$ and $\mathbf{c}$ are from the corpus.
- Maximise $P(X=0|\mathbf{w},\mathbf{c};\theta)$ if $\mathbf{w}$ and $\mathbf{c}$ are not from the corpus.

Therefore
$$
\begin{aligned} 
\theta &= \arg\max_{\theta}\prod_{(\mathbf{w},\mathbf{c})\in\mathcal{D}}P(X=1|\mathbf{w},\mathbf{c};\theta) \cdot \prod_{(\mathbf{w},\mathbf{c})\in\hat{\mathcal{D}}} P(X=0|\mathbf{w},\mathbf{c};\theta) \\
& = \arg\max_{\theta}\prod_{(\mathbf{w},\mathbf{c})\in\mathcal{D}}P(X=1|\mathbf{w},\mathbf{c};\theta) \cdot \prod_{(\mathbf{w},\mathbf{c})\in\hat{\mathcal{D}}} (1-P(X=1|\mathbf{w},\mathbf{c};\theta)) \\
& = \arg\max_{\theta}\sum_{(\mathbf{w},\mathbf{c})\in\mathcal{D}}\log P(X=1|\mathbf{w},\mathbf{c};\theta)+\sum_{(\mathbf{w},\mathbf{c})\in\hat{\mathcal{D}}} \log(1-P(X=1|\mathbf{w},\mathbf{c};\theta)) \\
& = \arg\max_{\theta}\sum_{(\mathbf{w},\mathbf{c})\in\mathcal{D}}\log \left(\frac{1}{1+\exp(-\mathbf{u}_w^\intercal\mathbf{v}_c)}\right)+\sum_{(\mathbf{w},\mathbf{c})\in\hat{\mathcal{D}}} \log\left(1-\frac{1}{1+\exp(-\mathbf{u}_w^\intercal\mathbf{v}_c)}\right) \\
& = \arg\max_{\theta}\sum_{(\mathbf{w},\mathbf{c})\in\mathcal{D}}\log \left(\frac{1}{1+\exp(-\mathbf{u}_w^\intercal\mathbf{v}_c)}\right)+\sum_{(\mathbf{w},\mathbf{c})\in\hat{\mathcal{D}}} \log\left(\frac{1}{1+\exp(\mathbf{u}_w^\intercal\mathbf{v}_c)}\right) \\
\end{aligned} 
$$
## CBOW Objective Function
Taking this approximation we can change the original objective function for CBOW,
$$J =-\mathbf{u}_c^\intercal\mathbf{\hat{v}}+\log\sum_{j=1}^{|V|}\exp(\mathbf{u}_j^\intercal\mathbf{\hat{v}})$$
into the following:
$$J=-\log\sigma(\mathbf{u}_c^\intercal\mathbf{\hat{v}})-\sum_{k=1}^{K}\log\sigma(-\hat{\mathbf{u}}_k^\intercal\mathbf{\hat{v}})$$
where,
- $\hat{\mathbf{u}}_k$ is embedding of a word in "not training corpus".
- $K$ is the total number of words in a sample.
## Skip-Gram Objective Function
We can do the same thing for the skip-gram objective function,
$$J=-\sum_{j=0,j\neq m}^{2m}\mathbf{u}_{c-m+j}^\intercal\mathbf{v}_c+2m\cdot\log\sum_{k=1}^{|V|}\exp(\mathbf{u}_k^\intercal\mathbf{v}_c)$$
So our objective function for a given centre word $c$ and context word $c-m+j$ will be:
$$J=-\log\sigma(\mathbf{u}_{c-m+j}^\intercal\mathbf{v}_c)-\sum_{k=1}^{K}\log\sigma(-\hat{\mathbf{u}}_k^\intercal\mathbf{v}_c)$$
where,
- $\hat{\mathbf{u}}_k$ is embedding of a word in "not training corpus".
- $K$ is the total number of words in a sample.
## Sampling From "Not Training Corpus"
### Unigram Distribution
Unigram distribution is a probability distribution where the words are distributed according their frequency occurrence, i.e. the chance of a word being negatively sampled is directly proportional to the number of times they appear in the training corpora. This is unlike **uniform distribution** where every word has an equal likelihood of being selected.

We have a set of numbers,
$$N(\mathbf{w})=[2,4,5,...,|V|\space numbers]$$
each representing the frequency of the word, corresponding to its index, in the entire corpus.

Finally, to get our distribution, we normalise each number in the set with the total number of words in the corpus.
$$Un(\mathbf{w})=\left[\frac{2}{\textnormal{len}(corpus)},\frac{4}{\textnormal{len}(corpus)},\frac{5}{\textnormal{len}(corpus)},...,|V|\space numbers \right]$$
### Power Law
According to the original paper, instead of just using the unigram distribution to sample from, we raise it to a power $3/4$.
$$P(\mathbf{w})=Un^{\frac{3}{4}}(\mathbf{w})$$
This increases rarer words being sampled more often and balances out the negative sampling process.

There is no theoretical explanation on why $3/4$ was chosen. It is mostly based on empirical results.

# References
- [CS224N Lecture Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4)