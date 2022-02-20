# Eye-Tracking
Check basem and gaze_track folders for code. cv_gaze.ipynb for training and some results!
Small (about 2M parameters) transformer-based model achieved 13.44% mean angular error on the MPIIGaze evaluation set while being only trained on UnityEyes.

### Aims
* ~~Add normal and fast MPIIGaze dataset code.~~
  
  Done, also added additional ipynb for mpiigaze.
* ~~Clean ipynb even more.~~
  
  Done, cleaned and added rubbish ipynb files, but now also need to add:
  * Visual testing in notebook.
* ~~Train pre-trained model on MPIIGaze.~~
  
  Done, achieved 9% angular error, which is not the best result, but can be used as a demo in the next steps. Later it can be improved by additional data, adding new layers as mentioned below and new training techniques (like mixup etc.). Also, maybe can add few-shot training, so if some data available, then good, if not then just procceed as usual.
* Production: Quantize, covert to tensorrt, try tempo (?) make a small server, etc (huggingface infinity?).
  
  Planning to use TensorRT (fp16 or int8) with torchserve (in docker container). Maybe, later, even more tools for better servicing and lifecycle. Will be done soon.

* Fix augmentation (maybe some new can be added) and re-train.
* Try the GELu activation unit instead of Swish.
* Different loss functions can be tested (L1?).
* More testing on different models, that are already written in basem (by me), but also other new architectures could be tried.
* Try TokenLearner.
* Something else that I forgot.

gelu, adjust transformation, add visual layers, add tokenize?

# Note
In the folder old_proj_gaze, you can find some previous works on the topic and the presentation.
