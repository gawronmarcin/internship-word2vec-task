import numpy as np

def sigmoid(x):
    x=np.clip(x,-500,500)
    return 1.0/(1.0+np.exp(-x))

text="""
cats like to eat fish. dogs like to eat meat. 
the hungry cat chases the small mouse. the big dog chases the cat. 
birds fly high in the blue sky. 
fish swim deep in the blue water.
birds and fish are animals. cats and dogs are animals.
"""
tokens=text.lower().split()
vocab=list(set(tokens))
vocab_size=len(vocab)

word2id={word: i for i, word in enumerate(vocab)}
idx2word = {i:word for i, word in enumerate(vocab)}

np.random.seed(1)

embedding_dim=10

W1=np.random.uniform(-0.1,0.1,(vocab_size,embedding_dim))
W2=np.random.uniform(-0.1,0.1,(vocab_size,embedding_dim))

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

epochs=50
learning_rate=0.05

for epoch in range(epochs):
    epoch_loss=0
    for target_id,context_id,label in training_data:
        v_c=W1[target_id]
        u_ctx=W2[context_id]
        dot_prod=np.dot(v_c,u_ctx)
        pred=sigmoid(dot_prod)

        eps=1e-10

        if label==1:
            loss=-np.log(pred+eps)
        else:
            loss=-np.log(1-pred+eps)

        epoch_loss+=loss

        error=pred-label

        grad_v_c=error*u_ctx
        grad_u_ctx=error*v_c
        W1[target_id]-=learning_rate*grad_v_c
        W2[context_id]-=learning_rate*grad_u_ctx

    if epoch%10==0:
        avg_error=epoch_loss/len(training_data)
        print(avg_error)
