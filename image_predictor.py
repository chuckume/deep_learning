from fastai.vision import *
from pathlib import Path

# Make sure that path contains the file 'export.pkl' from before.
path=Path('path_to_export.pkl_file')

learn = load_learner(path)

defaults.device = torch.device('cpu')

img = open_image(path/'AIGLE.png')
pred_class,pred_idx,outputs = learn.predict(img)
pred_class