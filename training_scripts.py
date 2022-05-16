from models.seg import train_seg_model, save_seg_model
from models.flair import train_flair_model, save_flair_model
from models.t1 import train_t1_model, save_t1_model
from models.t1ce import train_t1ce_model, save_t1ce_model
from models.t2 import train_t2_model, save_t2_model

def train_seg () :
    # Seg model
    seg = train_seg_model ()
    save_seg_model(seg, "./models/seg.pth")

def train_flair () :
    # flair model
    flair = train_flair_model ()
    save_flair_model(flair, "./models/flair.pth")

def train_t1 ():
    # t1 model
    t1 = train_t1_model ()
    save_t1_model(t1, "./models/t1.pth")

def train_t1ce ():
    # t1ce model
    t1ce = train_t1ce_model ()
    save_t1ce_model (t1ce, "./models/t1ce.pth")

def train_t2 ():
    # t2 model
    t2 = train_t2_model ()
    save_t2_model (t2, "./models/t2.pth")