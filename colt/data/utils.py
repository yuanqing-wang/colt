def scale(x):
    min, max = x.min(), x.max()
    return (x - min) / (max - min)