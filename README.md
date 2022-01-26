# Eye-Tracking
Check basem and gaze_track folders for code. cv_gaze.ipynb for training and some results!
Small (about 2M parameters) transformer-based model achieved 13.44% mean angular error on the MPIIGaze evaluation set while being only trained on UnityEyes.

### Aims
* Add normal and fast MPIIGaze dataset code.
* Clean ipynb even more.
* Train pre-trained model on MPIIGaze.
* Production: Quantize, covert to tensorrt, try tempo (?) make a small server, etc (huggingface infinity?).
* Fix augmentation (maybe some new can be added) and re-train.
* Try the GELu activation unit instead of Swish.
* Different loss functions can be tested (L1?).
* More testing on different models, that are already written in basem (by me), but also other new architectures could be tried.
* Try TokenLearner.
* Something else that I forgot.

# Note
In the folder old_proj_gaze, you can find some previous works on the topic and the presentation.
