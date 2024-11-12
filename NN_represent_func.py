import sys
import numpy as np
from sklearn.model_selection import train_test_split
from math import sqrt

import torch
from torch import nn
import torch.optim as optim


from sympy import symbols, sympify


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'



    

class inputvals(): # set default input values 
    def __init__(self, N : int = 100, test_size : float = 0.25, num_layr : int = 2, nneurons : int = 6, P : int = 1000, xmin : float = -2.0, xmax : float = 2.0, wnoise : str = 'y') -> None:
        self.N = N
        self.test_size = test_size
        self.num_layr = num_layr
        self.nneurons = nneurons
        self.P = P
        self.xmin = xmin
        self.xmax = xmax
        self.wnoise = wnoise
    def values(self):
        return [self.N,self.test_size,self.num_layr,self.nneurons,self.P,self.xmin,self.xmax,self.wnoise]
    
           




class NeuralNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int = 1,
        num_neurons: int = 5,
    ) -> None:
        """Basic neural network architecture with linear layers
        Args:
            num_layers (int, optional): number of hidden layers
            num_neurons (int, optional): neurons for each hidden layer
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        layers = []

        # input layer
        layers.append(nn.Linear(1, num_neurons))

        # hidden layers
        for _ in range(num_layers):
            layers.extend([nn.Linear(num_neurons, num_neurons), nn.Tanh()])

        # output layer
        layers.append(nn.Linear(num_neurons, 1))

        # build the network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.reshape(-1, 1)).squeeze()





def trainer(model,function,N,P,testsize,xmin,xmax,wnoise):
    x = symbols('x')
    xlist = np.linspace(float(xmin),float(xmax),N)
    Xtorch = torch.from_numpy(xlist).to(torch.float32).reshape(N,1)
    ylist = np.array([function.subs(x,xi) for xi in xlist])
    if wnoise == 'y':
        print('Add gaussian noise\n')
        ylist = ylist + np.random.normal(0,0.05,N)
    if wnoise == 'n':
        print('No gaussian noise added\n')
        ylist = ylist
    ylist = np.array(ylist,dtype=np.float32)
    Ytorch = torch.from_numpy(ylist)
    
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(Xtorch, Ytorch, test_size=testsize)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    running_loss = 0.0
    loss_time = []
    for epoch in range(0,P):
        optimizer.zero_grad()
        outputs = model(inputs_train)
        loss = criterion(outputs,targets_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loss_time = loss_time + [loss.item()]
        
    print('running loss = ',float(loss))
    loss_time = np.array(loss_time)
    epochlist = np.linspace(0,P,P)
    plt.plot(epochlist,loss_time,'--m')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    return [model,xlist,ylist]





def tester(model,N,xarr,yarr):
    xlist = xarr
    ylist = yarr
    xtorch = torch.from_numpy(xlist).to(torch.float32)
    tw = []
    for x in xtorch:
        w = x.reshape(1)
        tm = model(w).detach().numpy()
        tw = tw + [tm]
    
    tw = np.array(tw)
    diff = np.array([sqrt((tw[i]-ylist[i])**2) for i in range(0,len(tw))])

    plt.plot(xlist,ylist,'--r',label='function')
    plt.plot(xlist,tw,'--b',label='model')
    plt.plot(xlist,diff,'--g',label='loss')
    plt.axvline(x=xlist[0],color='c')
    plt.axvline(x=xlist[len(xlist)-1],color='c')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()




def inputer():
    sys.stdout.write('Choose Number of points:\n')
    N = input()
    sys.stdout.write('Choose the test size:\n')
    size = input()
    sys.stdout.write('Choose the number of layers:\n')
    nlayers = input()
    sys.stdout.write('Choose the number of neurons:\n')
    nneurons = input()
    sys.stdout.write('Choose the number of epochs:\n')    
    P = input()
    sys.stdout.write('Choose xmin:\n')
    xmin = input()
    sys.stdout.write('Choose xmax:\n')
    xmax = input()
    sys.stdout.write('With noise (y or n)?\n')
    wnoise = input()
    return [N,size,nlayers,nneurons,P,xmin,xmax,wnoise]




def model_builder(function,N,size,nlayers,nneurons,P,xmin,xmax,wnoise):
    N, nlayers, nneurons, size, P = int(N), int(nlayers), int(nneurons), float(size), int(P)
    model = NeuralNetwork(nlayers,nneurons)
    function = sympify(function) #convert str into function
    update = trainer(model,function,N,P,size,xmin,xmax,wnoise) # return model, xlist, ylist <- function
    tester(update[0],N,update[1],update[2])
    torch.save(model,'model')
    
    print(f"Train again?")
    while True:
        answer = input("\ny to train again or \nn to quit training\n")
        if answer.lower() not in ['y','n']:
            continue
        if answer.lower() == 'y':
            get = inputer() # input N, size, epochs, xmin and xmax
            return model_builder(function,get[0],get[1],get[2],get[3],get[4],get[5],get[6],get[7])
        if answer.lower() == 'n':
            print(f'Quit altogether or replot?\n')
            while True:
                answer = input('\ny to replot or \nn to exit\n')
                if answer.lower() == 'y':
                    oldmod = torch.load('model')
                    tester(oldmod,N,update[1],update[2])
                    continue
                if answer.lower() == 'n':
                    print("\n🎉🎉🎉🎉")
                    print("See you the next time!\n")
                    if __name__ == "__main__":
                        sys.exit(f"Bye! 👋")






    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit 1D function to a Neural Network"
    )



    parser.add_argument(
        '-nX','--ninputs', metavar='Npoints',
        help='The number of input points.'
    )

    parser.add_argument(
        '-s','--size', metavar='testsize',
        help='Size of the training set.'
    )

    parser.add_argument(
        '-p','--pepochs', metavar='Nepochs',
        help='Number of epochs for the training.'
    )

    parser.add_argument(
        '-f','--function', metavar='expression',
        required=True, help='The function you want to probe'
    )
    

    parser.add_argument(
        '-x0','--xmin', metavar='interval_min',
        help='The xmin in the interval'
    )

    parser.add_argument(
        '-x1','--xmax', metavar='interval_max',
        help='The xmax in the interval'
    )



    parser.add_argument(
        '-nl','--nlayers', metavar='Nlayers',
        help='The number of layers'
    )

    parser.add_argument(
        '-nn','--nneurons', metavar='Nneurons',
        help='The number of neurons'
    )


    parser.add_argument(
        '-wn','--wnoise', metavar='add_noise',
        help='Logical, w/wo noise'
    )
    
    args = parser.parse_args()

   
    default = inputvals().values()
        
    if args.ninputs == None:
        ninputs = default[0]
    else:
        ninputs = args.ninputs
        
    if args.size == None:
        size = default[1]
    else:
        size = args.size

    if args.nlayers == None:
        nlayers = default[2]
    else:
        nlayers = args.nlayers

    if args.nneurons == None:
        nneurons = default[3]
    else:
        nneurons = args.nlayers

    if args.pepochs == None:
        pepochs = default[4]
    else:
        pepochs = args.pepochs

    if args.xmin == None:
        xmin = default[5]
    else:
        xmin = args.xmin

    if args.xmax == None:
        xmax = default[6]
    else:
        xmax = args.xmax
    if args.wnoise == None:
        noise = default[7]
    else:
        noise = args.wnoise
    

    print(f"Welcome to the Matrix! 🤖")

    print('N points =',ninputs)
    print('test size = ',size)
    print('number of layers = ',nlayers)
    print('number of neurons = ',nneurons)
    print('number of epochs = ',pepochs)
    print('xmin = ',xmin)
    print('xmax = ',xmax)
    print('with noise = ',noise)

    
    model_builder(args.function,ninputs,size,nlayers,nneurons,pepochs,xmin,xmax,noise)
    
