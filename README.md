# embed2spec
Code is messy, sorry for that in advance.
In multiscale_convolution.py, you can find two of the models. We've tried various others but those two are the main models we've mostly used. Training code for multiscale model is also in this file.
In training.py, you can find training code for simpler convolutional model (named CNN) that is defined in multiscale_convolution.py
In discriminator.py, there is discriminator model and training code for GAN structure. It uses multiscale convolution decoder as generator.
ModelEvaluator.py is the class that produces audio given the trained model.
dataset.py consists of two dataset generators one with speaker information and one without.
preprocess_fft.py takes log stft of dataset and stores it. 

There are few points that are bugging me so that you can take a look at them first and tell me if there is something wrong. 
1) Since each voice active region differs in duration, resulting embeddings are also different in size. i.e. (64, 120) vs (64, 230)
Thus, while passing them through to the network we need to crop or pad them in some manner. We've tried various possible ways: 
  a) Detect the longest embedding sequence and repeat other shorter ones to match to the longest sequence. For example, if the longest embedding is (64,860), we repeated all the shorter ones along axis 1 so that they can match in size. 
  b) Detect the shortest embedding sequence and crop all the others. For example, if the shortest sequence is (64,100), we only took first 100 samples. 
  c) Take the mean of the all sequences and crop or repeat according to length of each sequence.
I don't know if we could make it any other way, but somehow, I think this might be a problem.
2) While preproccessing (computing stft), we use 512 point FFT, 512 point window length (corresponds to 32 ms) and 160 point hop length (corresponds to 10 ms). As far as I know, frames that are embeddings obtained from were 30 ms with 10 ms shift. Murat hoca suggested using a window with length that is power of 2. But I'm not sure if this causes any alignment issues. It sounds like it should. Tonight I will regenerate stfts and see if it makes a difference. Just wanted to take your opinion about it. 
