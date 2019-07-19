import numpy as np
import os

i = 0
for fname in os.listdir(os.getcwd()):
    i += 1
    if fname.endswith(".npz"):
        print('processing ' + fname + '...')
        
        base_name = fname.split('.')[0]

        # load npz file
        data = np.load(fname)

        # frequency bins
        fbins = np.array([4000*2**((2*i+1)/32) for i in range(96)])

        # electron noise
        e_noise = data['e_noise']

        # data to save
        save_data = np.transpose(np.array([fbins, e_noise]))
        
        # name of the saved file
        save_name = base_name + '.txt'

        # save to file
        np.savetxt(save_name, save_data, fmt = '%1.4e',
                   header = 'f(Hz)\t Electron noise power spectrum density (V^2/Hz)')
