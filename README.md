## Dynamic Dropout

```
git clone https://github.com/kushalj001/dynamic-dropout.git
conda create -n dynamic python=3.7
pip install -r requirements.txt
// for 3080 Ti machines:
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
export CUBLAS_WORKSPACE_CONFIG=:4096:8
// change the project folder in computation.py
```

We're currently testing two new ideas:
### How does cell dropout affect autoregressive decoding?

Most of the current and past work only apply dropout on the hidden vector of the LSTM layers.
Our aim is to gain more fine-grained control of the hidden and cell states that are passed across the time-steps (across words in a sentence) during autoregressive decoding.   
To this end, we use LSTM cells for AR decoding instead of `nn.LSTM`. This gives us access to the hidden and cell states. There isn't any work/paper that analyzes how the information encoded in the cell states helps AR decoding. We conjecture that cell states carry important information across timesteps and hence applying dropout on them would force the model to extract more information from the latent vector. We have 2 layers of LSTM cells unrolled across the sequence length. The hope is that dropping out neurons from the first layer amounts to dropping information at a word level and dropping out from the second layer is equivalent to reduce information from text spans, which essentially makes the dropout hierarchical in nature.
The current implementation simply drops out neurons from hidden and cell equally across all time-steps. 
Make sure to uncomment or add `do_not_use_double_lstm` in the training config to use this setting. 

### How does replacing LSTM encoder with a transformer encoder affect the latent space and metrics?

Hyperparameters added to control the transformer encoder:
```
'--use_transformer_encoder', action='store_true'
'--nheads', type=int, default=8
'--transformer_dropout', type=float, default=0.2
'--transformer_ffdim', type=int, default=2048 
'--num_transformer_blocks', type=int, default=1
'--transformer_activation', type=str, default="relu" 
```
Transformer layer is a single transformer layer with 1 self-attention layer, a feed-forward layer and layer norm. We can stack such layers using `num_transformer_blocks`, which is by default 1. The other hyperparams are self-explanatory. The input dimension to the transformer layer is 512 which is the same as embedding dimension. This is essentially the model dimension for the transformer which means that all the computations within the transformer layer happens on 512-dim vectors for each token. The hidden dimension in LSTM encoder was 2048. It's not clear how these 2 dimensions correlate as of now but the `d_model` can be increased if results are not satisfactory. That would also require a change in the embedding dimension of the word encoder.