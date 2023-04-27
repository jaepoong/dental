from .base_model import *

def load_Generator(args):
    if args.model=="base":
        model=BaseGenerator(n_res=args.n_res)
    
    if args.resize:
        model=Resizing_Generator(n_res=args.n_res) # current tackle only 256 size
    
    if args.gray:
        model=Gray_Generator(n_res=args.n_res)
    
    if args.gray and not args.resize:
        model=NonResizing_GrayGenerator(n_res=args.n_res)
    return model

def load_Discriminator(args):
    if args.model=="base":
        model=BaseDiscriminator()
    
    if args.gray:
        model=GrayDiscriminator()
    return model