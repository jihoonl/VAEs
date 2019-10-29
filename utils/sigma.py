

def get_sigma(epoch):
    if epoch < 20:
        return 1.0
    elif 20 < epoch < 40:
        return 0.5
    else:
        return 0.1
