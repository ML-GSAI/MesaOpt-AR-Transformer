from .lsa import AutoregressiveLSA


def create_model(args):
    """
    Returns suitable model from its name.
    Arguments: args
    Returns:
        torch.nn.Module.
    """
    if args.model == 'lsa':
        model = AutoregressiveLSA(dim=args.dim, T=args.T, a=args.a, b=args.b)
    else:
        raise ValueError('Invalid model name {}!'.format(args.model))
    
    return model
