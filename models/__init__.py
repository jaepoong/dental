from .base_model import *

def load_Generator(args):
    if args.model=="base":
        model=BaseGenerator(n_res=args.n_res)
    if args.model=="U_Net":
        model=U_Net()
    if args.model=="AttU_Net":
        model=AttU_Net()

    return model

def load_Discriminator(args):
    model=BaseDiscriminator()
    
    return model