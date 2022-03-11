"""
Multilayer perceptron model based on the Energy efficient synaptic plasticity paper(https://elifesciences.org/articles/50804)
Made by Balazs Agoston Nyiro, 2022
University of Nottingham
"""

from tqdm import tqdm 
from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as cp #if you have nvidia GPU, write cupy instead of numpy
from sklearn.model_selection import train_test_split
import time
import copy
import matplotlib.pyplot as plt
import pandas as pd
import csv
from csv import DictReader

class Data(object):
    '''
    1. download the MNIST dataset
    2. normalize
    3. make train and test datas in random order    
    '''

  def __init__(self):
      x_data, y_data = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
      x_data = (x_data/255).astype('float32')
      y_data = to_categorical(y_data)
      test_ratio = 0.14285714285714285714285714285714
      self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data, test_size=test_ratio, random_state=42)
      pass

class Multilayer_Perceptron():
    def __init__(self, hid_unit_num, max_epoch, learn_rate, finish_mode, decay_mode, decay_rate, energy_scale_maintenance, req_acc, cons_mode, cons_th, saving_freq, consolidation_freq):
        self.hid_unit_num = hid_unit_num
        self.max_epoch = max_epoch
        self.learn_rate = learn_rate
        self.learning_time = 1
        self.finish_mode = finish_mode
        self.permanent = {} #just to transfer IN&OUT values between forward and backward prop.
        self.decay_mode = decay_mode
        self.decay_rate = decay_rate
        self.energy_scale_maintenance = energy_scale_maintenance
        self.trans_energy = 0
        self.cons_energy = 0
        self.perc_energy = 0
        self.required_acc = req_acc
        self.cons_mode = cons_mode
        self.cons_treshold = cons_th
        if self.cons_mode == "no_cons":
            self.consolidation_freq = 1
        else:
            self.consolidation_freq = consolidation_freq
        
        self.saving_freq = saving_freq
        self.consolidation_freq = consolidation_freq
        
        # save paramaters into a dictionary
        self.parameters_initial, self.parameters_persistent, self.parameters_transient, self.parameters_history, self.permanent = self.initialization()


    def __get_total_weights__(self):
        '''
        It gives back: total memory = initial - (transient + persistent)
        '''
        WT1_id = cp.add(self.parameters_persistent['WT1'], self.parameters_transient['WT1'])
        WT2_id = cp.add(self.parameters_persistent['WT2'], self.parameters_transient['WT2'])
        WT1_sum = cp.subtract(self.parameters_initial['WT1'], WT1_id)
        WT2_sum = cp.subtract(self.parameters_initial['WT2'], WT2_id)
        
        total_weights  = {
            'WT1' : WT1_sum,
            'WT2' : WT2_sum
        }

        return total_weights
    

    def __set_total_weights__(self):
        '''
        1. Total memory = initial - (transient + persistent)
        2. Refresh permanent memory
        '''
        
        WT1_id = cp.add(self.parameters_persistent['WT1'], self.parameters_transient['WT1'])
        WT2_id = cp.add(self.parameters_persistent['WT2'], self.parameters_transient['WT2'])
        WT1_sum = cp.subtract(self.parameters_initial['WT1'], WT1_id)
        WT2_sum = cp.subtract(self.parameters_initial['WT2'], WT2_id)
        self.permanent['WT1'] = WT1_sum
        self.permanent['WT2'] = WT2_sum


    def sigmoid_function(self, x, derivate=False):
        '''
        sigmoid function for the hidden layer
        '''
        if derivate:
            return (cp.exp(-x))/((cp.exp(-x)+1)**2)
        return 1/(1 + cp.exp(-x))

    def synaptic_decay(self):
        '''
        Exponential decay with decay rate. new_weights = old_weights*(1-decay_rate)^predictation_timestep. It only does if decay mode is True.
        '''
        if self.decay_mode and self.cons_mode != "no_cons":
            self.parameters_transient['WT1'] *= cp.power(cp.exp(-self.decay_rate),1)
            self.parameters_transient['WT2'] *= cp.power(cp.exp(-self.decay_rate),1)
        else:
            pass

    def softmax_function(self, x, derivate=False):
        '''
        softmax function for the output layer
        '''
        expons = cp.exp(x - x.max())
        if derivate:
            return expons / cp.sum(expons, axis=0) * (1 - expons / cp.sum(expons, axis=0))
        return expons / cp.sum(expons, axis=0)

    def trans_energy_calc(self):
        '''
        this function calculate the transient energy. If there is no consolidation the transient energy = 0
        '''
        if self.cons_mode == "no_cons":
            self.trans_energy += self.energy_scale_maintenance * ( cp.sum( cp.abs(self.parameters_transient['WT1'])) + cp.sum( cp.abs(self.parameters_transient['WT2'])) )
        else:
            self.trans_energy += 0.0 * ( cp.sum( cp.abs(self.parameters_transient['WT1'])) + cp.sum( cp.abs(self.parameters_transient['WT2'])) )

    def min_energy_calc(self):
        '''
        minimum energy = sum(|wi(current)-wi(initial)|)
        '''
        current_weights = self.__get_total_weights__()
        difference_WT1 = current_weights['WT1'] - self.parameters_initial['WT1']
        difference_WT2 = current_weights['WT2'] - self.parameters_initial['WT2']
        min_energy = cp.sum( cp.abs(difference_WT1)) + cp.sum( cp.abs(difference_WT2))

        return min_energy

    def cons_energy_calc(self, weight_difference):
        '''
        this funcion calculate the consoldation energy
        '''
        self.cons_energy += cp.sum( cp.abs(weight_difference['WT1'])) + cp.sum( cp.abs(weight_difference['WT2']))  


    def consolidation(self):
        '''
        it makes the consolidation
        '''
        
        old_weights = copy.deepcopy(self.parameters_persistent) #make a copy of the persistent memory before the consolidation

        if self.cons_mode == "no_cons":  #if there is no consolidation all of the transient weights are added to the persistent weights
            self.parameters_persistent['WT1'] += self.parameters_transient['WT1']
            self.parameters_persistent['WT2'] += self.parameters_transient['WT2']
            self.parameters_transient['WT1'] = cp.zeros((self.hidden_layer, self.input_layer))
            self.parameters_transient['WT2'] = cp.zeros((self.output_layer, self.hidden_layer))

       
        elif self.cons_mode == "lc_th_lc_cons": #only those weights will be added to the persistent weights which are over the consolidation threshold
          for dict_indexes, dict_params in self.parameters_transient.items():
              for array_indexes, cp_arrays in enumerate(dict_params):
                  for val_indexes, values in enumerate(cp_arrays):
                    if cp.abs(values)>=self.cons_treshold: 
                      self.parameters_persistent[dict_indexes][array_indexes][val_indexes] += self.parameters_transient[dict_indexes][array_indexes][val_indexes]
                      self.parameters_transient[dict_indexes][array_indexes][val_indexes] = 0.00
        
        elif self.cons_mode == "lc_th_gl_cons": #if one weight is over the threshold, every weights will be added to the persistent weights
          count_above_th = 0
          for dict_indexes, dict_params in self.parameters_transient.items():
              for array_indexes, cp_arrays in enumerate(dict_params):
                  for val_indexes, values in enumerate(cp_arrays):
                    if cp.abs(values)>=self.cons_treshold:
                      count_above_th += 1

          if count_above_th>0:
              self.parameters_persistent['WT1'] += self.parameters_transient['WT1']
              self.parameters_persistent['WT2'] += self.parameters_transient['WT2']
              self.parameters_transient['WT1'] = cp.zeros((self.hidden_layer, self.input_layer))
              self.parameters_transient['WT2'] = cp.zeros((self.output_layer, self.hidden_layer))


        elif self.cons_mode == "gl_th_gl_cons": # all the weight should be over the threshold
          count_above_th = 0
          count_len_dict = 0
          for dict_indexes, dict_params in self.parameters_transient.items():
              for array_indexes, cp_arrays in enumerate(dict_params):
                  for val_indexes, values in enumerate(cp_arrays):
                    count_len_dict += 1                    
                    if cp.abs(values)>=self.cons_treshold:
                      count_above_th += 1

          if count_above_th == count_len_dict:
              self.parameters_persistent['WT1'] += self.parameters_transient['WT1']
              self.parameters_persistent['WT2'] += self.parameters_transient['WT2']
              self.parameters_transient['WT1'] = cp.zeros((self.hidden_layer, self.input_layer))
              self.parameters_transient['WT2'] = cp.zeros((self.output_layer, self.hidden_layer))
        

        elif self.cons_mode_ != "lc_th_lc_cons" or self.cons_mode_ != "lc_th_gl_cons" or self.cons_mode_ != "gl_th_gl_cons" :
            raise ValueError("Not existing consoldation mode")
        else:
            raise ValueError("Value problem in the consolidation function")

        # calculate the weight change of the persistent memory
        weight_difference = {
            'WT1' : self.parameters_persistent['WT1'] - old_weights['WT1'],
            'WT2' : self.parameters_persistent['WT2'] - old_weights['WT2']
        }

        self.cons_energy_calc(weight_difference)


    def initialization(self):
        '''
        Set number of nodes in layers and weight matrixes.
        '''

        self.input_layer = 784
        self.hidden_layer = self.hid_unit_num
        self.output_layer = 10

        parameters_initial = {
            'WT1':cp.random.randn(self.hidden_layer, self.input_layer)* cp.sqrt( 1. / self.hidden_layer), 
            'WT2':cp.random.randn(self.output_layer, self.hidden_layer)* cp.sqrt( 1. / self.output_layer)
        }

        parameters_persistent = {
            'WT1':cp.zeros((self.hidden_layer, self.input_layer)),
            'WT2':cp.zeros((self.output_layer, self.hidden_layer))
        }

        parameters_transient = {
            'WT1':cp.zeros((self.hidden_layer, self.input_layer)),
            'WT2':cp.zeros((self.output_layer, self.hidden_layer))
        }

        permanent = {}

        parameters_history = {
            'time': [],
            'epoch' : [],
            'accuracy' : [],
            'min_energy' : [],
            'con_energy' : [],
            'trans_energy': [],
            'perceptron_energy': [],
            'total_energy' : [],
            'parameters' : []
        }

        return parameters_initial, parameters_persistent, parameters_transient, parameters_history, permanent

    def forward_propagation(self, x_train):
        '''
        This is the forward propagation algorithm, for calculating the updates of 
        the neural network's parameters in the forward direction. It uses sigmoid
        funcion at the hidden layer and softmax at the output layer.
        '''
        
        self.__set_total_weights__()
        parameters_total = self.permanent

        # icput layer activations becomes sample
        parameters_total['OUT_0'] = x_train

        # icput layer to hidden layer 1
        parameters_total['IN_1'] = cp.dot(parameters_total['WT1'], parameters_total['OUT_0'])
        parameters_total['OUT_1'] = self.sigmoid_function(parameters_total['IN_1'])

        # hidden layer 2 to output layer
        parameters_total['IN_2'] = cp.dot(parameters_total['WT2'], parameters_total['OUT_1'])
        parameters_total['OUT_2'] = self.softmax_function(parameters_total['IN_2'])

        return parameters_total['OUT_2']

    def backpropagation(self, y_train, output):
        '''
        This is the backpropagation algorithm, for calculating the updates of 
        the neural network's parameters in the backward direction.
        '''

        parameters_total = self.permanent
        weight_change = {}

        # Calculate weight changes between the output and the hidden layer
        error = 2 * (output - y_train) / output.shape[0] * self.softmax_function(parameters_total['IN_2'], derivate=True)
        weight_change['WT2'] = cp.outer(error, parameters_total['OUT_1'])

        # Calculate weight changes between the hidden layer and the icput layer
        error = cp.dot(parameters_total['WT2'].T, error) * self.sigmoid_function(parameters_total['IN_1'], derivate=True)
        weight_change['WT1'] = cp.outer(error, parameters_total['OUT_0'])

        return weight_change


    def update_network(self, w_change):
        '''
        Update network parameters according to Stochastic Gradient Descent
        '''

        for ind, val in w_change.items():
            self.parameters_transient[ind] += self.learn_rate * val

       
    def compute_accuracy(self, x_test, y_test):
        '''
        This function performs a forward pass of x, then checks if the indices 
        of the output's highest value match the indices in the label y. After 
        that, it adds together all of the prediction and calculates the accuracy.
        '''

        pred = cp.array([])

        for x__test_data, y__test_data in zip(x_test, y_test):
            output = self.forward_propagation(x__test_data)
            ind_pred = cp.argmax(output)
            pred = cp.append(pred,ind_pred == cp.argmax(y__test_data))
            #pred.append(ind_pred == cp.argmax(y__test_data))
      
        return cp.mean(pred)


    def train(self, x_train, y_train, x_test, y_test):
        '''
        This functions run through the epochs and call the forward, backward propagation
        and the weight update functions. After each epocs it calculates an accuracy.
        '''
        
        start_time = time.time()
        for iter in range(self.max_epoch):
            print()
            print(f"Processing epoch: {iter+1}/{self.max_epoch} --> ", end = ' ')
            for ind_train_pic,ind_train_label in tqdm(zip(x_train, y_train), total = x_train.shape[0]):
                self.learning_time +=1
                output = self.forward_propagation(ind_train_pic)
                w_change = self.backpropagation(ind_train_label, output)
                self.update_network(w_change)


                if(((self.learning_time+1)%self.consolidation_freq)==0):

                    self.consolidation()
                    self.trans_energy_calc()
                    self.synaptic_decay()

                else: 

                    self.trans_energy_calc()
                    self.synaptic_decay()


                if(((self.learning_time+1)%self.saving_freq)==0):

                    min_energy = self.min_energy_calc()
                    total_energy = self.trans_energy + self.cons_energy
                    batch_accuracy = self.compute_accuracy(x_test, y_test)

                    self.parameters_history['time'].append(self.learning_time)
                    self.parameters_history['epoch'].append(iter)
                    self.parameters_history['accuracy'].append(float(batch_accuracy))
                    self.parameters_history['min_energy'].append(float(min_energy))
                    self.parameters_history['con_energy'].append(float(self.cons_energy))
                    self.parameters_history['trans_energy'].append(float(self.trans_energy))
                    self.parameters_history['total_energy'].append(float(total_energy))
                    self.parameters_history['perceptron_energy'].append(float(self.perc_energy))

            accuracy = self.compute_accuracy(x_test, y_test)
            print("\033[94m Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}% \033[0m".format(
                iter+1, time.time() - start_time, accuracy * 100
            ))
            
            if batch_accuracy >= self.required_acc and self.finish_mode == "success_rate": 
                        return batch_accuracy,total_energy
            elif batch_accuracy < self.required_acc and self.finish_mode == "till_the_end":
                        pass            
            
        return batch_accuracy,total_energy

def write_hist_to_csv(histories):
    '''
    this function will save the histroies log dictionary as a csv 
    '''
    
    headers = [ 'time', 'epoch', 'accuracy', 'min_energy', 'con_energy', 'trans_energy', 'perceptron_energy', 'total_energy']
    filename = 'figure_' + str(FIGURE) + '_' + time.strftime("%Y-%m-%d_%H%M%S") + '.csv'

    keys = histories[0].keys()

    a_file = open(filename, "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(histories)
    a_file.close()


def plotting(xdata, ydata, xaxis_name, yaxis_name, label_names, plottitle = False, xlim = False, ylim = False, yscale = 'log'):
    '''
    Makes the line graphs.
    '''
        
    filename = 'figure_' + str(FIGURE) + '_' + time.strftime("%Y-%m-%d_%H%M%S") + '.png'
    
    #iterate through the lines of the line graphs and makes the plot
    for ind, val in enumerate(xdata):        
        plt.plot(xdata[ind], ydata[ind], label = label_names[ind])
    
    if xlim != False : plt.xlim(xlim)
    if ylim != False : plt.ylim(ylim)
    if plottitle != False : plt.title(plottitle)
    plt.yscale(yscale)
    plt.legend()
    plt.xlabel(xaxis_name)
    plt.ylabel(yaxis_name)
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()
    


def make_plots(figure_name):
        '''
        This function contains all of the paramaters to make the individual figures. 
        1. set the initial parameters
        2. run the learning
        3. save the histories dictionary into csv
        4. make the plot
        '''

        if figure_name  == '5A':
            
            print("\033[1m Run learning for Figure 5A... \033[0m")
            print('')
            
            MAX_EPOCH = 100 #normaly =100
            HIDDEN_LAYER_UNITT_NUM = 100 #normaly = 100
            learning_rate_array = [0.001, 0.01, 0.1, 0.5] # normaly = [0.001, 0.01, 0.1, 0.5]
            FINISH_MODE =  "success_rate" # 1.till_the_end 2. success_rate
            DECAY_MODE = False # True if there is a decay, and False if there is no decay
            DECAY_RATE = 0.001 #normaly = 0.001
            ENERGY_SCALE_MAINTENANCE = 0.001 #normaly = 0.001
            REQUIRED_ACCURACY = 0.93 #normaly = 0.96
            CONSOLIDATION_MODE = "no_cons"  #normaly = no_cons
            CONSOLIDATION_TRESHOLD = 0.00005 #normaly =0.0005
            SAVING_FREQ = 10000
            CONSOLIDATION_FREQ = 1000

            histories = []
            
            for LEARNING_RATE in learning_rate_array:
                
                initial_parameters = [HIDDEN_LAYER_UNITT_NUM, MAX_EPOCH, LEARNING_RATE, FINISH_MODE, DECAY_MODE, DECAY_RATE, ENERGY_SCALE_MAINTENANCE, REQUIRED_ACCURACY, CONSOLIDATION_MODE, CONSOLIDATION_TRESHOLD, SAVING_FREQ, CONSOLIDATION_FREQ]
                mlp = Multilayer_Perceptron(HIDDEN_LAYER_UNITT_NUM, 
                                            MAX_EPOCH, 
                                            LEARNING_RATE, 
                                            FINISH_MODE, 
                                            DECAY_MODE, 
                                            DECAY_RATE, 
                                            ENERGY_SCALE_MAINTENANCE, 
                                            REQUIRED_ACCURACY, 
                                            CONSOLIDATION_MODE, 
                                            CONSOLIDATION_TRESHOLD,
                                            SAVING_FREQ,
                                            CONSOLIDATION_FREQ
                                            )
                
                mlp.parameters_history['parameters'] = initial_parameters 
                print("\033[1m Perceptron initialization done ==> " + f"Max epoch: {MAX_EPOCH} | Number of hidden units: {HIDDEN_LAYER_UNITT_NUM} | Learning rate: {LEARNING_RATE} | Consolidation mode: {CONSOLIDATION_MODE} | Stop accuracy: {REQUIRED_ACCURACY}  \033[0m")
                mlp.train(cp.array(dat.x_train), cp.array(dat.y_train), cp.array(dat.x_test), cp.array(dat.y_test))
                histories.append(mlp.parameters_history)
            
            print("\033[1m Learning done. Save weights... \033[0m")
            print('')
            write_hist_to_csv(histories)
            
            print("\033[1m creating plot for Figure 5A... \033[0m")
            print('')
            xdata = []
            ydata = []
            label_names = []
            plottitle = False
            xlim = False
            ylim = False
            xaxis_name = 'Accuracy'
            yaxis_name = 'Energy'
            yscale = "log"

                       
            for ind, val in enumerate(histories):
                label_names.append( 'n=' + str(learning_rate_array[ind]) )
                ydata.append( cp.dot(val['total_energy'],1) )
                xdata.append( cp.dot(val['total_energy'],1) )
                        
            plotting(xdata, ydata, xaxis_name, yaxis_name, label_names, plottitle, xlim, ylim , yscale)
            
            print("\033[1m Figure 5A done \033[0m")
            print('')

        elif figure_name == '5B':
            
            print("\033[1m Run learning for Figure 5B... \033[0m")
            print('')
            
            MAX_EPOCH = 30 #normaly =30
            HIDDEN_LAYER_UNITT_NUM = 100 #normaly = 100
            LEARNING_RATE = 0.1
            FINISH_MODE =  "success_rate" # 1.till_the_end 2. success_rate
            DECAY_MODE = True # True if there is a decay, and False if there is no decay
            DECAY_RATE = 0.001 #normaly = 0.001
            ENERGY_SCALE_MAINTENANCE = 0.001
            REQUIRED_ACCURACY = 0.96
            cons_mode_array = ['no_cons', 'lc_th_lc_cons']
            CONSOLIDATION_TRESHOLD = 0.00005 #normaly =0.0005
            SAVING_FREQ = 10000
            CONSOLIDATION_FREQ = 1000

            histories = []
            
            for CONSOLIDATION_MODE in cons_mode_array:
                initial_parameters = [HIDDEN_LAYER_UNITT_NUM, MAX_EPOCH, LEARNING_RATE, FINISH_MODE, DECAY_MODE, DECAY_RATE, ENERGY_SCALE_MAINTENANCE, REQUIRED_ACCURACY, CONSOLIDATION_MODE, CONSOLIDATION_TRESHOLD, SAVING_FREQ, CONSOLIDATION_FREQ]
                mlp = Multilayer_Perceptron(HIDDEN_LAYER_UNITT_NUM, 
                                            MAX_EPOCH, 
                                            LEARNING_RATE, 
                                            FINISH_MODE, 
                                            DECAY_MODE, 
                                            DECAY_RATE, 
                                            ENERGY_SCALE_MAINTENANCE, 
                                            REQUIRED_ACCURACY, 
                                            CONSOLIDATION_MODE, 
                                            CONSOLIDATION_TRESHOLD,
                                            SAVING_FREQ,
                                            CONSOLIDATION_FREQ
                                            )
                
                mlp.parameters_history['parameters'] = initial_parameters 
                print("\033[1m Perceptron initialization done ==> " + f"Max epoch: {MAX_EPOCH} | Number of hidden units: {HIDDEN_LAYER_UNITT_NUM} | Learning rate: {LEARNING_RATE} | Consolidation mode: {CONSOLIDATION_MODE} | Stop accuracy: {REQUIRED_ACCURACY}  \033[0m")
                mlp.train(cp.array(dat.x_train), cp.array(dat.y_train), cp.array(dat.x_test), cp.array(dat.y_test))
                histories.append(mlp.parameters_history)
            
            print("\033[1m Learning done. Save weights... \033[0m")
            print('')
            
            write_hist_to_csv(histories)
            
            print("\033[1m Creating plot for Figure 5B... \033[0m")
            print('')
            
            xdata = []
            ydata = []
            label_names = []
            plottitle = False
            xaxis_name = 'Accuracy'
            yaxis_name = 'Energy'
            ylim = (1000, 10000000)
            xlim = (0.8, 0.96)
            yscale = "log"
            
            
            label_names.append('Without synaptic caching')
            xdata.append(histories[0]['accuracy'])
            ydata.append(cp.dot(histories[0]['total_energy'],1))
            label_names.append('With synaptic caching')
            xdata.append(histories[0]['accuracy'])
            ydata.append(cp.dot(histories[1]['total_energy'],1))
            label_names.append('Minimum energy')
            xdata.append(histories[0]['accuracy'])
            ydata.append(cp.dot(histories[1]['min_energy'],1))

            
            plotting(xdata, ydata, xaxis_name, yaxis_name, label_names, plottitle, xlim, ylim, yscale)
            print("\033[1m Figure 5B done. \033[0m")
            print('')
            
        elif figure_name == '5C':
            
            print("\033[1m Run learning for Figure 5C... \033[0m")
            print('')
            
            MAX_EPOCH = 80 #normaly =200
            hidden_layer_num = list(cp.linspace(1, 200, 30, dtype=int))
            LEARNING_RATE = 0.1
            FINISH_MODE =  "success_rate" # 1.till_the_end 2. success_rate
            DECAY_MODE = True # True if there is a decay, and False if there is no decay
            DECAY_RATE = 0.001 #normaly = 0.001
            ENERGY_SCALE_MAINTENANCE = 0.001
            required_aquracy_array = [0.85, 0.93]
            cons_mode_array = ['no_cons', 'lc_th_lc_cons']
            CONSOLIDATION_TRESHOLD = 0.0005 #normaly =0.01
            SAVING_FREQ = 10000
            CONSOLIDATION_FREQ = 1000

            histories = []
            hid_layer_per_energy_histories = []
            
            for CONSOLIDATION_MODE in cons_mode_array:
                for REQUIRED_ACCURACY in required_aquracy_array:
                    
                    hist_of_total_energy_all_hidd_layer = []
                    hist_of_hidnum_all_hidd_layer = []
                    
                    for HIDDEN_LAYER_UNITT_NUM in hidden_layer_num:
                        
                        initial_parameters = [HIDDEN_LAYER_UNITT_NUM, MAX_EPOCH, LEARNING_RATE, FINISH_MODE, DECAY_MODE, DECAY_RATE, ENERGY_SCALE_MAINTENANCE, REQUIRED_ACCURACY, CONSOLIDATION_MODE, CONSOLIDATION_TRESHOLD, SAVING_FREQ, CONSOLIDATION_FREQ]
                        mlp = Multilayer_Perceptron(HIDDEN_LAYER_UNITT_NUM, 
                                            MAX_EPOCH, 
                                            LEARNING_RATE, 
                                            FINISH_MODE, 
                                            DECAY_MODE, 
                                            DECAY_RATE, 
                                            ENERGY_SCALE_MAINTENANCE, 
                                            REQUIRED_ACCURACY, 
                                            CONSOLIDATION_MODE, 
                                            CONSOLIDATION_TRESHOLD,
                                            SAVING_FREQ,
                                            CONSOLIDATION_FREQ
                                            )
                        
                        mlp.parameters_history['parameters'] = initial_parameters 
                        print("\033[1m Perceptron initialization done ==> " + f"Max epoch: {MAX_EPOCH} | Number of hidden units: {HIDDEN_LAYER_UNITT_NUM} | Learning rate: {LEARNING_RATE} | Consolidation mode: {CONSOLIDATION_MODE} | Stop accuracy: {REQUIRED_ACCURACY}  \033[0m")
                        return_accuracy, return_total_energy = mlp.train(cp.array(dat.x_train), cp.array(dat.y_train), cp.array(dat.x_test), cp.array(dat.y_test))
                        histories.append(mlp.parameters_history)
                        
                        if return_accuracy < REQUIRED_ACCURACY: 
                            pass #raise ValueError(f"The learnign didn't achieved the required accuracy. Set the max epoch higher and try again")
                        else: 
                            hist_of_total_energy_all_hidd_layer.append(return_total_energy)
                            hist_of_hidnum_all_hidd_layer.append(HIDDEN_LAYER_UNITT_NUM)
                            
                    hid_layer_per_energy_histories.append([hist_of_hidnum_all_hidd_layer,hist_of_total_energy_all_hidd_layer])
                    
            
            print("\033[1m Learning done. Save weights... \033[0m")
            
            write_hist_to_csv(histories)
            
            print("\033[1m Creating plot for Figure 5C... \033[0m")
            print('')
            
            xdata = []
            ydata = []
            label_names = []
            xaxis_name = '#hidden units'
            yaxis_name = 'Energy'
            plottitle = False
            xlim = False
            ylim = (1000, 10000000)
            yscale = "log"
            
            
            label_names.append('No caching, 85% accuracy')
            xdata.append(hid_layer_per_energy_histories[0][0])
            ydata.append(hid_layer_per_energy_histories[0][1])
            label_names.append('No caching, 93% accuracy')
            xdata.append(hid_layer_per_energy_histories[1][0])
            ydata.append(hid_layer_per_energy_histories[1][1])
            label_names.append('Synaptic caching, 85% accuracy')
            xdata.append(hid_layer_per_energy_histories[2][0])
            ydata.append(hid_layer_per_energy_histories[2][1])
            label_names.append('Synaptic caching, 93% accuracy')
            xdata.append(hid_layer_per_energy_histories[3][0])
            ydata.append(hid_layer_per_energy_histories[3][1])

            plotting(xdata, ydata, xaxis_name, yaxis_name, label_names, plottitle, xlim, ylim, yscale)
            
            print("\033[1m Figure 5C done. \033[0m")
            print('')
            
        elif figure_name == '4B':
            
            print("\033[1m Run learning for Figure 4B... \033[0m")
            print('')
            
            MAX_EPOCH = 30 #normaly =200
            HIDDEN_LAYER_UNITT_NUM = 100
            LEARNING_RATE = 0.1
            FINISH_MODE =  "till_the_end" # 1.till_the_end 2. success_rate
            DECAY_MODE = False # True if there is a decay, and False if there is no decay
            DECAY_RATE = 0.001 #normaly = 0.001
            ENERGY_SCALE_MAINTENANCE = 0.001
            REQUIRED_ACCURACY = 0.93
            cons_mode_array = ['no_cons', 'lc_th_lc_cons', 'lc_th_gl_cons', 'gl_th_gl_cons']
            CONSOLIDATION_TRESHOLD = 0.000005 #normaly =0.01
            SAVING_FREQ = 10000
            CONSOLIDATION_FREQ = 1000

            histories = []
            hid_layer_per_energy_histories = []
            
            for CONSOLIDATION_MODE in cons_mode_array:
                        
                        initial_parameters = [HIDDEN_LAYER_UNITT_NUM, MAX_EPOCH, LEARNING_RATE, FINISH_MODE, DECAY_MODE, DECAY_RATE, ENERGY_SCALE_MAINTENANCE, REQUIRED_ACCURACY, CONSOLIDATION_MODE, CONSOLIDATION_TRESHOLD, SAVING_FREQ, CONSOLIDATION_FREQ]
                        mlp = Multilayer_Perceptron(HIDDEN_LAYER_UNITT_NUM, 
                                            MAX_EPOCH, 
                                            LEARNING_RATE, 
                                            FINISH_MODE, 
                                            DECAY_MODE, 
                                            DECAY_RATE, 
                                            ENERGY_SCALE_MAINTENANCE, 
                                            REQUIRED_ACCURACY, 
                                            CONSOLIDATION_MODE, 
                                            CONSOLIDATION_TRESHOLD,
                                            SAVING_FREQ,
                                            CONSOLIDATION_FREQ
                                            )
                        
                        mlp.parameters_history['parameters'] = initial_parameters 
                        print("\033[1m Perceptron initialization done ==> " + f"Max epoch: {MAX_EPOCH} | Number of hidden units: {HIDDEN_LAYER_UNITT_NUM} | Learning rate: {LEARNING_RATE} | Consolidation mode: {CONSOLIDATION_MODE} | Stop accuracy: {REQUIRED_ACCURACY}  \033[0m")
                        mlp.train(cp.array(dat.x_train), cp.array(dat.y_train), cp.array(dat.x_test), cp.array(dat.y_test))
                        histories.append(mlp.parameters_history)

            print("\033[1m Learning done. Save weights... \033[0m")
            
            write_hist_to_csv(histories)
            
            print("\033[1m Creating plot for Figure 4B... \033[0m")
            print('')
            
            xdata = []
            ydata = []
            label_names = []
            xaxis_name = 'Cost of transient energy'
            yaxis_name = 'Energy'
            plottitle = False
            xlim = False
            ylim = False
            yscale = "log"
            
            label_names.append('Without synaptic caching')
            xdata.append(histories[0]['trans_energy'])
            ydata.append(cp.dot(histories[0]['total_energy'],1))
            label_names.append('Local treshold, local consolidation')
            xdata.append(histories[1]['trans_energy'])
            ydata.append(cp.dot(histories[1]['total_energy'],1))
            label_names.append('Local treshold, global consolidation')
            xdata.append(histories[2]['trans_energy'])
            ydata.append(cp.dot(histories[2]['total_energy'],1))
            label_names.append('Gobal treshold, global consolidation')
            xdata.append(histories[3]['trans_energy'])
            ydata.append(cp.dot(histories[3]['total_energy'],1))
            label_names.append('Minimum energy')
            xdata.append(histories[0]['trans_energy'])
            ydata.append(cp.add(histories[0]['min_energy'],0.000000001))

            plotting(xdata, ydata, xaxis_name, yaxis_name, label_names, plottitle, xlim, ylim, yscale)
            
            print("\033[1m Figure 4B done. \033[0m")
            print('')
                                    
        else:
            raise ValueError("This figure hasn't programed yet")


'''
ONLY CHANGE THIS PART
'''

required_figures = ['4B','5A'] # WRITE THE NAME(S) OF REQUIRED GRAPH(S) : 4B, 5A , 5B or 5C

print("\033[1m Data initialization in progress... \033[0m") #download the dataset and generate the learnign and test datas
dat = Data()
print('Done')
print('')

# iterate through the required figures
for FIGURE in required_figures:
  print(FIGURE)
  make_plots(FIGURE)