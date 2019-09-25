from fastai.vision import *
from pathlib import Path

# Make sure that path contains the file 'export.pkl' from before.
#path=Path('path_to_export.pkl_filelearn = load_learner(path)

defaults.device = torch.device('cpu')

img = open_image(path/'AIGLE.png')
pred_class,pred_idx,outputs = learn.predict(img)
print( {"prediction": pred_class.__dict__['obj'],
        "confidence": outputs.numpy()[pred_class.__dict__['data'].item()]*100})

