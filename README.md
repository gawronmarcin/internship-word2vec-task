# internship-word2vec-task

I'll implement the word2vec training loop on a simple text using the skip-gram with a negative sampling method. 

I'll begin by importing the NumPy library and defining a sigmoid function. It's a function that transforms values into 0-1 range, making them into probabilities. I'll also clip the values as to avoid overflow while exponentiating.

```python
import numpy as np

def sigmoid(x):
    x=np.clip(x,-500,500)
    return 1.0/(1.0+np.exp(-x))
```
Now I'll create a list of unique words in my text sample 

```python
text="""
cats like to eat fish. dogs like to eat meat. 
the hungry cat chases the small mouse. the big dog chases the cat. 
birds fly high in the blue sky. 
fish swim deep in the blue water.
birds and fish are animals. cats and dogs are animals.
"""
text = text.replace('.', '')
tokens=text.lower().split()
vocab=list(set(tokens))
vocab_size=len(vocab)
```

I'll be creating matrices with words as rows, so it would be useful to give each word an index (so it's possible to get an index from a word and the other way around). Two dictionaries in both ways will help with that. 
```python
word2id={word: i for i, word in enumerate(vocab)}
idx2word = {i:word for i, word in enumerate(vocab)}
```
Now I can create two matrices. One for vectors for words as targets and another for words as context. In the end, the first one will be the one with vectors that matter. I'll fill the matrices with random values from -0.1 to 0.1 (so the sigmoid doesn't transform them into only zeroes and ones, which would make the training unstable and all over the place). I'll also set the random seed to 1 to get comparable results each time the training is done. 
```python
np.random.seed(1)

embedding_dim=10

W1=np.random.uniform(-0.1,0.1,(vocab_size,embedding_dim))
W2=np.random.uniform(-0.1,0.1,(vocab_size,embedding_dim))
```
Now I have to make the stage ready for training, which means I have to prepare pairs of words and label whether they really are near each other or whether they're a false pair. I'll look at 4 real context words for each word,  and for each one of them, I'll take 3 random negative samples.
```python
 window_size=2
num_negative_samples=3

training_data=[]

for i in range(len(tokens)):
    target_id=word2id[tokens[i]]
    start=max(0,i-window_size)
    end=min(len(tokens),i+window_size+1)

    true_context_indices = [word2id[tokens[k]] for k in range(start, end) if k != i]

    for j in range(start,end):
        if i!=j:
            context_id=word2id[tokens[j]]
            training_data.append((target_id,context_id,1))
            for _ in range(num_negative_samples):
                neg_id=np.random.randint(0,vocab_size)
                while neg_id==target_id or neg_id in true_context_indices:
                    neg_id=np.random.randint(0,vocab_size)
                training_data.append((target_id, neg_id, 0))
```
I'm also checking if the negative sample really isn't in the vicinity of the targeted word As far as I understand, this can be skipped while working on a large dataset, because the probability of getting a "false negative-sample" is so low that it can be dismissed as noise, but my dataset is small, which is why I implemented this kind of safeguard. 



And now comes the most important part - the training itself. I'm going to set a number of epochs, one epoch is going through the entire dataset once. I'll also set the learning_rate to 0.05. Learning rate is the rate of how aggressively model changes the weights. I'm also defining epoch loss variable, which I'll use to print average loss, so I can see if the model is getting better with epoch.
```python
epochs=50
learning_rate=0.05

for epoch in range(epochs):
    epoch_loss=0
    for target_id,context_id,label in training_data:
        v_c=W1[target_id]
        u_ctx=W2[context_id]

        #Forward Pass
        dot_prod=np.dot(v_c,u_ctx)
        pred=sigmoid(dot_prod)

        #Loss
        eps=1e-10

        if label==1:
            loss=-np.log(pred+eps)
        else:
            loss=-np.log(1-pred+eps)
        epoch_loss+=loss
        #Gradients
        error=pred-label

        grad_v_c=error*u_ctx
        grad_u_ctx=error*v_c

        #Parameter updates
        W1[target_id]-=learning_rate*grad_v_c
        W2[context_id]-=learning_rate*grad_u_ctx
    if epoch%10==0:
      avg_error=epoch_loss/len(training_data)
      print(avg_error)
```
In the forward pass, I'm calculating the dot product of the target and context vectors. The sigmoid function returns a probability value where the closer the number is to 1, the closer the meaning of the words. 

I'm calculating Loss with the Binary Cross-Entropy function, which is the standard for comparing predicted probabilities to true labels. Due to the use of logarithms, it gives a huge error to wildly inaccurate guesses and a small error to guesses close to the truth. 

To calculate gradients, so the values by which I'll make vectors from both matrices closer or further away, I need to calculate the error. Error is just a derivative of the BCE function, and it actually simplifies to prediction - true label. 

After calculating the gradients, they can be used to update the parameters and make vectors closer or further away. 

And that's the whole word2vec training loop. After going through all epochs, the first matrix with "target" vectors gives each word vectors that can be used to define how close in meaning they are to other words in the matrix. 
