def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "lwf":
        from models.lwf import LwF
        return LwF(args)
    elif name == "wa":
        from models.wa import WA
        return WA(args)
    elif name == "finetune":
        from models.finetune import Finetune
        return Finetune(args)
    elif name == "replay":
        from models.replay import Replay
        return Replay(args)
    elif name == "foster":
        from models.foster import FOSTER
        return FOSTER(args)
    elif name == "memo":
        from models.memo import MEMO
        return MEMO(args)
    elif name == "nccil":
        from models.nccil import NCCIL
        return NCCIL(args)
    elif name == "upcl":
        from models.upcl import UPCL_iCaRL
        return UPCL_iCaRL(args)
    elif name == "cosinebaseline":
        from models.upcl import CosineBaseline
        return CosineBaseline(args)
    else:
        assert 0
