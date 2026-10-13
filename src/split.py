from collections import defaultdict 


def scaffold_split(df, frac_train=0.8, frac_valid=0.1):
    """Group molecules by scaffold; largest groups fill train first,
    so valid/test contain smaller, structurally rarer scaffolds."""
    groups = defaultdict(list)
    for idx, scaf in enumerate(df["scaffold"]):
        groups[scaf].append(idx)
        
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    
    n = len(df)
    train_cut = frac_train * n 
    valid_cut = (frac_train + frac_valid) * n 
    
    train, valid, test = [], [], []
    for g in ordered:
        if len(train) + len(g) <= train_cut:
            train.extend(g)
        elif len(train) + len(valid) + len(g) <= valid_cut:
            valid.extend(g)
        else:
            test.extend(g)
    return train, valid, test 