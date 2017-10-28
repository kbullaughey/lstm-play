---
layout: post
title:  Summary of Facebook Voice Loop paper
---

Here I discuss [Voice Synthesis for in-the-Wild Speakers via a Phonological Loop][1], which is a recent paper out of Facebook's AI group.

[1]: https://arxiv.org/abs/1707.06588

This paper offers a neural text-to-speech model that is remarkable in how well it performs for such a simple architecture. Also, the architecture itself is interesting, as it has this shifting buffer that the authors refer to as a phonological loop. To cap if off, they provide code and trained models and used publicly available data for training. All of these factors make it an ideal playground for TTS experiments.

### Background

This past year has been a busy year for advances in neural text-to-speech.

- 2017 October - [Voice Loop][2] (single speaker) <br><audio src="https://bullaughey.com/audio/tts-examples/loop-single/1.wav" controls></audio>
- 2017 October - [Deep Voice 3][3] <br><audio src="https://bullaughey.com/audio/tts-examples/deep-voice-3/1.wav" controls></audio>
- 2017 October - [WaveNet][6] (fast) <br><audio src="https://bullaughey.com/audio/tts-examples/google/Hol_After.wav" controls></audio>
- 2017 August - [Siri][11] (iOS 11) <br><audio src="https://bullaughey.com/audio/tts-examples/siri/ios11b.wav" controls></audio>
- 2017 July - [Voice Loop][2] (multi speaker, tuned to Donald Trump's voice) <br><audio src="https://bullaughey.com/audio/tts-examples/loop-multi/trump.wav" controls></audio>
- 2017 May - [Deep Voice 2][7] <br><audio src="https://bullaughey.com/audio/tts-examples/deep-voice-2/loose-weight.wav" controls></audio>
- 2017 March - [Tacotran][4] <br><audio src="https://bullaughey.com/audio/tts-examples/tacotron/gan_or_vae_r2.wav" controls></audio>
- 2017 February - [Char2Wav][8] <br><audio src="https://bullaughey.com/audio/tts-examples/char2wav/original_best_bidirectional_encoder_6.wav" controls></audio>
- 2017 February - [Deep Voice 1][9] <br><audio src="https://bullaughey.com/audio/tts-examples/deep-voice-1/change-the-world.wav" controls></audio>
- 2016 November - [AWS Polly][10] <br><audio src="https://bullaughey.com/audio/tts-examples/aws-polly/1.wav" controls></audio>
- 2016 September - [WaveNet][5] (slow) <br><audio src="https://bullaughey.com/audio/tts-examples/google/Hol_Before.wav" controls></audio>

[2]: https://ytaigman.github.io/loop/site/
[3]: http://research.baidu.com/deep-voice-3-2000-speaker-neural-text-speech/
[4]: https://google.github.io/tacotron/publications/tacotron/index.html
[5]: https://deepmind.com/blog/wavenet-generative-model-raw-audio/
[6]: https://deepmind.com/blog/wavenet-launches-google-assistant/
[7]: http://research.baidu.com/deep-voice-2-multi-speaker-neural-text-speech/
[8]: http://josesotelo.com/speechsynthesis/
[9]: http://research.baidu.com/deep-voice-production-quality-text-speech-system-constructed-entirely-deep-neural-networks/
[10]: https://aws.amazon.com/polly/details/
[11]: https://machinelearning.apple.com/2017/08/06/siri-voices.html

Before neural TTS methods, the most typical approaches were:

0. Unit selection (concatenative), whereby a database of phones or partial phones is mined for optimal context and spliced together. This example is from [Festival][12] (Alan):<br><audio src="https://bullaughey.com/audio/tts-examples/festival/concat/1.wav" controls></audio>
0. Parametric synthesis, whereby an HMM or other model is trained and then used to synthesize speech. This example is also from [Festival][12] (SLT HTS 2011 voice):<br><audio src="https://bullaughey.com/audio/tts-examples/festival/hmm/1.wav" controls></audio>

[12]: http://www.cstr.ed.ac.uk/projects/festival/morevoices.html

### Inputs

Once the model is trained, the only input is the raw text. This is converted via a lookup table based on the CMU dictionary to phonemes. The sequence of phoneme identities are then fed into neural net.

### Outputs

The outputs of the model are vocoder parameters, not raw waveforms. The training data is pre-processed to estimate the vocoder parameters and these are used as the ground truth that the model is trained against. A vocoder vector is 63 dimensions and corresponds to a 10ms frame of audio. [Merlin][13] is then used to synthesize waveforms based on these vocoder frames.

[13]: https://github.com/CSTR-Edinburgh/merlin

### Training Data

The model was trained using the [VCTK corpus][14]. The full corpus includes 109 voices, but for this project only the 21 American English voices were used. Here are some more stats regarding training data:

0. **400** sentences read by each voice.
0. **8** hours of audio in aggregate.
0. **207,559** phonemes in aggregate.
0. **3,037,337** vocoder frames in aggregate.
0. **10** ms long vocoder frames.
0. **8015** example sentences in aggregate.

[14]: http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html

### Validation Data

They use a validation set to assess the model. This includes all the same 21 American voices (all represented in both training and validation).
352 of the 424 sentences are represented.
There are no sentences in the validation set that are not represented in the training data by at least one other voice.
There are no examples (voice/sentence combos) in the validation set that are shared with the training set (so it's legit).

### Architecture

#### Attention network

They use an attention model pulled directly from the Graves paper used for generating handwriting. This attention model predicts parameters of a Gaussian mixture model. The means are transformed to (0,∞) and used as an offset. This forces the alignment to be monotonic. Interestingly, this attention model was instead used for prediction because the handwriting RNN couldn't capture sufficiently long temporal dependencies. This was before this type of temporal weighting was even called attention (a term coined the following year).

In this paper the attention model isn't used in a prediction network, instead it is used to summarize the inputs into a context vector, that is used as an input to the loop buffer update.
More specifically, the weights that come out of the attention model are used to produce an affine combination of the input embeddings (i.e. over the sequence of phonemes).

Attention model summary:

0. Inputs: vectorized loop buffer.
0. Outputs: GMM mixture priors, means, and variances.
0. Model structure: Simple, fully-connected feed-forward network with 1 hidden layer.

#### Buffer update network

The loop buffer and how this buffer is updated is the crux of their model, and probably the most innovative and important part. 

The loop buffer is a matrix with columns corresponding to phone-input positions. At each vocoder timestep, the loop buffer contents are shifted right one column, discarding the right-most column and updating the first column with the output of the buffer-update network. (I suppose they refer to it as a loop because it could have been implemented as a circular buffer and instead of shifting columns, they could have updated the head position.)

The loop buffer is represented as a matrix for the purpose of shifting columns and performing the update (i.e., one column is updated). But whenever the loop buffer is used as an input to one of the neural networks, it is used as one long, unrolled vector.

Buffer update summary:

0. Inputs:
    0. The vectorized buffer
    0. Context vector (attention-weighted and combined input embeddings)
    0. Previous output
0. Output:
    0. The new first-column of the next time step's loop buffer.
    0. The projected voice embedding is mixed into network buffer update output.
0. Model structure: Simple feed forward network with 1 hidden layer.
0. Buffer shifts right one slot each timestep.

#### Output network
Input is the vectorized loop buffer.
Simple feed forward network with 1 hidden layer.
Projected voice embedding is mixed into output.

### Multiple voices

The model is trained on 21 different voices. Each voice is represented by learned embedding. This embedding is used in two places.

In both cases a linear transform of the voice embedding is:

0. mixed in to the loop buffer update that is output from the buffer update network..
0. mixed in to the vocoder parameter estimates that are output by the output network.

When initially training on the 21 voices selected from the VCTK corpus, the model learns how to compactly represent different voice qualities, textures, and reading styles. The hope is that this representation and the way the model uses that representation is general and can be applied to new voices.

They demonstrate that this is indeed the case by learning new voice embeddings after the model has been trained. To do this they freeze all the other parameters except the new voice embedding (a 256-vector) and train just those weights with new audio data from the new voice. These voice samples comprise "10s of minutes" for each new voice and come from Youtube videos that were transcribed by an automatic speech recognition system (presumably the one built into Youtube). The authors point out that these audio samples are much less uniform and more noisy than the original corpus, and include things like clapping and occasional other speakers.

### Training

They train on about 8 hours of audio from 21 speakers. Inputs are phones and outputs are vocoder frame parameters.

They use a fuzzy type of teacher forcing whereby instead of using the true output from the previous timestep (like normal teacher forcing), they use a combination of three things:

0. The ground truth from the previous timestep.
0. The predicted output from the previous timestep.
0. A noise vector.

They say this was necessary to get the model to train well.

I tried out training the model to benchmark it. Here are some comments:

0. You need to use pytorch version 1.1.12, the latest version (0.2.0) has breaking API changes.
0. As it stands, the code requires a GPU.
0. On an AWS p3.2xlarge instance (1 Nvidia Tesla V100 with > 5000 cores) running the new Nvidia GPU Cloud Amazon AMI I observed the following:
    0. Loading the data into GPU memory took 12 minutes.
    0. Required 2.2GB of GPU memory.
    0. Utilized the GPU at about 30% capacity, varying from 10% to 80% utilization.
    0. Each epoch takes an average of 9 minutes and they recommend training for 180 epochs in the README.
    0. Thus, I estimate total training time is about 25-30 hours.
0. The trained model they provide for download has 12,764,500 parameters.

Because the model is used to predict vocoder parameters and not actual waveforms, there is a limit to how natural the audio can sound. They are actually getting rather close to how well they can theoretically do given the final waveforms are produced by a parametric vocoder.

Here I provide three columns of audio:

0. Speech reconstructed from the ground truth vocoder parameters (the best they can expect to do).
0. Speech reconstructed from the vocoder parameters predicted by the model.
0. The true waveform that is part of the VCTK corpus.

All of these samples are out-of-sample (i.e., from the validation set).

<table>
 <thead>
  <tr>
  <th>Truth, vocoded</th>
  <th>Modeled, vocoded</th>
  <th>Original waveform</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/out-of-sample/p294_012.gen_0.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/out-of-sample/p294_012.orig.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/vctk/downsampled/p294_012.wav"></audio></td>
  </tr>
  <tr>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/out-of-sample/p305_017.gen_5.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/out-of-sample/p305_017.orig.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/vctk/downsampled/p305_017.wav"></audio></td>
  </tr>
  <tr>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/out-of-sample/p308_220.gen_7.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/out-of-sample/p308_220.orig.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/vctk/downsampled/p308_220.wav"></audio></td>
  </tr>
 </tbody>
</table>

It's worth noting that the in sample audio sounds noticeably better, meaning the model is perhaps over-fitting or not generalizing well enough. Perhaps they need more than 25 minutes of audio per voice. Here is a similar table to above, but in-sample:

<table>
 <thead>
  <tr>
  <th>Truth, vocoded</th>
  <th>Modeled, vocoded</th>
  <th>Original waveform</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/in-sample/p301_286.gen_4.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/in-sample/p301_286.orig.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/vctk/downsampled/p301_286.wav"></audio></td>
  </tr>
  <tr>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/in-sample/p305_052.gen_5.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/in-sample/p305_052.orig.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/vctk/downsampled/p305_052.wav"></audio></td>
  </tr>
  <tr>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/in-sample/p311_324.gen_9.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/voice-loop/results/in-sample/p311_324.orig.wav"></audio></td>
   <td><audio controls src="https://bullaughey.com/audio/vctk/downsampled/p311_324.wav"></audio></td>
  </tr>
 </tbody>
</table>


### Discussion

What is most remarkable about this paper is how well a trained model can be adapted with minimal additional data to highly recognizable produce speech from a new voice. The [demo samples][15] they published include synthetic speech for four well-known individuals, Donald Trump, Barak Obama, Mark Zuckerberg, and Sheryl Sandberg. This model is impressive because it never saw data from those voices until the 10s of minutes it was given later, and it still produced highly individualized, recognizable speech, and did so with an incredibly simple architecture and implementation.

[15]: https://ytaigman.github.io/loop/site/

Many of the architectural details of this paper have antecedents in the 2013 Graves paper. These include:

0. They directly use the Graves attention model. 
0. The voice embeddings are a bit similar to how the handwriting model is seeded to produce a new sample, or a particular style.

Unlike the Graves 2013 paper, they use simple feed forward networks rather than RNNs.

The authors claim that their approach is less sensitive to messy data. For fitting new voices (e.g., Donald, Obama, Mark, and Sheryl) they used Youtube videos which include clapping, occasionally other speakers, background noise, and varying recording environments, equipment and sampling rates. Generally people have not been able to train good TTS models on such data. Even attempts to train Tacotron on audiobook data has generally gone badly because the voice actor modulates their voice for different speaking parts and the reading is more dramatic with more complex timing and stress patterns. But it's worth keeping in mind that the authors only trained the speaker embeddings on this messy data. When fitting a new speaker the rest of the model parameters were kept fixed. I think it's thus a stretch to suggest their model is as robust as they claim to conditions in the wild.


