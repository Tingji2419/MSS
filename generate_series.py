from utils import set_seed
import numpy as np
import random
from numpy.lib.stride_tricks import sliding_window_view

def constant_signal(am=None, length=120, freq=0.02, noise=True):
    am = random.randint(-10, 10) / 10 if not am else am
    if noise:
        return am * np.ones(length) + np.random.normal(0, 1, length) * 0.05
    return am * np.ones(length)

def linear_signal(am=None, length=120, freq=0.02, noise=True):
    am = random.randint(-5, 5) / 10 if not am else am
    am = am / length
    cm = random.randint(-5, 5) / 10

    x = np.arange(0, length * freq, freq)
    y = am * (x)  + cm

    # if noise:
    #     return y + np.random.normal(0, 1, length) * 0.005
    return y

def stair_signal(am=None, length=120, freq=0.02, noise=True):
    am = random.randint(-5, 5) / 100 if not am else am
    bm = random.randint(30, length) // 10
    x = np.arange(0, bm) * am
    
    x = np.repeat(x, length // bm + 1)

    return x[:length] + np.random.normal(0, 1, length) * 0.05

def sawtooth_signal(am=None, length=120, freq=0.02, noise=True):
    am = random.randint(-5, 5) / 10 if not am else am
    bm = random.randint(30, length) // 10
    x = np.arange(0, bm) * am
    
    x = np.concatenate([x] * (length // bm + 1))

    return x[:length] + np.random.normal(0, 1, length) * 0.05

def square_signal(am=None, length=120, freq=0.02, noise=True):
    am = random.randint(-5, 5) / 10 if not am else am
    bm = random.randint(30, length) // 4
    x = np.ones(length) * am
    for index in range(length):
        if (index // bm) % 2 == 0:
            x[index] = -am

    return x + np.random.normal(0, 1, length) * 0.05


def sin_signal(am=None, length=120, freq=0.02, noise=True):
    am = random.randint(-10, 10) / 10 if not am else am
    if noise:
        noise = np.random.normal(0, 1, length)
        return am * np.sin(np.arange(0, length * freq, freq) * 5) + noise * 0.05
    
    return am * np.sin(np.arange(0, length * freq, freq) * 5)

def sin_signal_mix_sawtooth(am=None, length=120, freq=0.02, noise=True):

    return sin_signal(am, length, freq, noise) * sawtooth_signal(0.01, length, freq, noise)

def sin_signal_mix_square(am=None, length=120, freq=0.02, noise=True):

    return sin_signal(am, length, freq, noise) * square_signal(am, length, freq, noise)

def sin_signal_mix_cos(am=None, length=120, freq=0.02, noise=True):

    return sin_signal(am, length, freq, noise) * cos_signal(0.01, length, freq, noise)

def cos_signal(am=None, length=120, freq=0.02, noise=True):
    am = random.randint(-10, 10) / 10 if not am else am
    if noise:
        noise = np.random.normal(0, 1, length)
        return am * np.sin(np.arange(0, length * freq, freq) * 5) + noise * 0.05
    
    return am * np.sin(np.arange(0, length * freq, freq) * 5)

# def cos_signal_mix(am=None, length=120, freq=0.02, noise=True):
#     return cos_signal(am, length, freq, noise) * sawtooth_signal(am, length, freq, noise)

SIGNAL = {
    'constant': constant_signal,
    # 'constant2': constant_signal,
    'linear': linear_signal,
    'sawtooth': sawtooth_signal,
    'stair': stair_signal,
    'sin': sin_signal,
    'cos': cos_signal,
    'square': square_signal,
    'sin2': sin_signal_mix_sawtooth,
    # 'cos2': cos_signal_mix,
    'sin3': sin_signal_mix_square,
    'sin4': sin_signal_mix_cos,
}


set_seed(42)

class Artifical_Signal():
    '''
    Generate artifical signals to train discriminant models
    '''
    def __init__(self, num=5000, windows_size=250) -> None:
        self.num = num
        self.ws = windows_size
        self.cps = None
        self.series = np.zeros(self.num + self.ws)
        
        self.generate_change_points()


    def generate_change_points(self):
        '''
        Function: Generate num/2 change points to make balance 
        Frequency: 50Hz
        '''
        ratio_y = 0.5 
        min_action = int(self.ws) * 1
        
        if self.cps == None:
            self.cps = sorted(random.sample(list(np.arange(0, self.num, min_action)), int(ratio_y *self.num / min_action)))
        return self.cps

    def generate_signal(self):
        # 换成连续一条，然后滑动窗口
        signal_choices = [i for i in SIGNAL.values()]
        for pre, cur in zip(self.cps[:-1], self.cps[1:]):
            cur_signal = random.choice(signal_choices)
            self.series[pre: cur] += cur_signal(am=None, length=cur-pre, freq=0.02, noise=True)
        
        labels = np.zeros(self.num + self.ws)
        labels[self.cps] = 1
        self.labels = [int(sum(i) > 0) for i in sliding_window_view(labels, self.ws)]

        return sliding_window_view(self.series, self.ws)[1:], self.labels[1:]
    #     self.generate_change_points()

    #     choices = [i for i in SIGNAL.values()]
    #     for index, i in enumerate(self.cps):
    #         self.series[index * self.ws: index * self.ws + i] = random.choice(choices)(am=None, length=self.ws, freq=0.02, noise=True)
    #         self.series[index * self.ws + i + 1:] = random.choice(choices)(am=None, length=self.es, freq=0.02, noise=True)
        
    #     for index, _ in range(self.num / 2):
    #         self.series[index * self.ws: (index + 1) * self.ws] = random.choice(choices)(am=None, length=self.ws, freq=0.02, noise=True)
                
    #     return self.series, self.cps

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # a = linear_signal(am=None, length=120, freq=0.02, noise=True)
    # plt.plot(a)
    # plt.savefig('2.png')
    dataset = Artifical_Signal(num=50000, windows_size=200)
    X, y = dataset.generate_signal()

    
    print('Totoal:', len(y))
    print('1 in label:', sum(y) / len(y))
    plt.plot(dataset.series[:30000])

    plt.savefig('1.png')
