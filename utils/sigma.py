

def get_sigma(epoch, switch=3):

    if switch == 1:
        if epoch < 20:
            return 1.0
        elif 20 < epoch < 40:
            return 0.5
        else:
            return 0.1
    elif switch == 2:
        if epoch < 20:
            return 0.1
        elif 20 < epoch < 40:
            return 0.5
        else:
            return 1.0
    elif switch == 3:
        return 0.1
    else:
        raise NotImplemented(f'Switch {switch} not supported')
