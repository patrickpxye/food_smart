from __future__ import print_function

"""
Extra set of callbacks.
"""

import random
import warnings
import numpy as np
import logging
from keras.callbacks import Callback as KerasCallback
from keras_wrapper.utils import decode_predictions_one_hot, decode_predictions_beam_search, decode_predictions, decode_multilabel

import evaluation
from read_write import *


def checkDefaultParamsBeamSearch(params):
    required_params = ['model_inputs', 'model_outputs', 'dataset_inputs', 'dataset_outputs']
    default_params = {'beam_size': 5,
                      'maxlen': 30,
                      'normalize': False,
                      'alpha_factor': 1.0,
                      'words_so_far': False,
                      'n_parallel_loaders': 5,
                      'optimized_search': False,
                      'temporally_linked': False,
                      'link_index_id': 'link_index',
                      'state_below_index': -1,
                      'pos_unk': False,
                      'heuristic': 0,
                      'mapping': None
                      }

    for k, v in params.iteritems():
        if k in default_params.keys() or k in required_params:
            default_params[k] = v

    for k in required_params:
        if k not in default_params:
            raise Exception('The beam search parameter ' + k + ' must be specified.')

    return default_params


###################################################
# Performance evaluation callbacks
###################################################

class EvalPerformance(KerasCallback):
    def __init__(self,
                 model,
                 dataset,
                 gt_id,
                 metric_name,
                 set_name,
                 batch_size,
                 each_n_epochs=1,
                 extra_vars=None,
                 is_text=False,
                 is_multilabel=False,
		 multilabel_idx=None,
                 min_pred_multilabel=0.5,
                 index2word_y=None,
                 input_text_id=None,
                 index2word_x=None,
                 sampling='max_likelihood',
                 beam_search=False,
                 write_samples=False,
                 save_path='logs/performance.',
                 reload_epoch=0,
                 eval_on_epochs=True,
                 start_eval_on_epoch=0,
                 is_3DLabel=False,
                 write_type='list',
                 sampling_type='max_likelihood',
                 save_each_evaluation=True,
                 out_pred_idx=None,
                 max_plot=1.0,
                 verbose=1):
        """
        Evaluates a model each N epochs or updates

        :param model: model to evaluate
        :param dataset: instance of the class Dataset in keras_wrapper.dataset
        :param gt_id: identifier in the Dataset instance of the output data to evaluate
        :param metric_name: name of the performance metric
        :param set_name:  name of the set split that will be evaluated
        :param batch_size: batch size used during sampling
        :param each_n_epochs: sampling each this number of epochs or updates
        :param extra_vars: dictionary of extra variables
        :param is_text: defines if the predicted info is of type text (in that case the data will be converted from values into a textual representation)
        :param is_multilabel: are we applying multi-label prediction?
        :param multilabel_idx: output index where to apply the evaluation (set to None if the model has a single output)
        :param min_pred_multilabel: minimum prediction value considered for positive prediction
        :param index2word_y: mapping from the indices to words (only needed if is_text==True)
        :param input_text_id:
        :param index2word_x: mapping from the indices to words (only needed if is_text==True)
        :param sampling: sampling mechanism used (only used if is_text==True)
        :param beam_search: whether to use a beam search method or not
        :param write_samples: flag for indicating if we want to write the predicted data in a text file
        :param save_path: path to dumb the logs
        :param reload_epoch: reloading epoch
        :param eval_on_epochs: eval each epochs (True) or each updates (False)
        :param start_eval_on_epoch: only starts evaluating model if a given epoch has been reached
        :param is_3DLabel: defines if the predicted info is of type 3DLabels
        :param write_type:  method used for writing predictions
        :param sampling_type: type of sampling used (multinomial or max_likelihood)
        :param save_each_evaluation: save the model each time we evaluate (epochs or updates)
        :param out_pred_idx: index of the output prediction used for evaluation (only applicable if model has more than one output, else set to None)
        :param max_plot: maximum value shown on the performance plots generated
        :param verbose: verbosity level; by default 1
        """
        self.model_to_eval = model
        self.ds = dataset
        self.gt_id = gt_id
        self.input_text_id = input_text_id
        self.index2word_x = index2word_x
        self.index2word_y = index2word_y
        self.is_text = is_text
        self.is_multilabel = is_multilabel
        self.multilabel_idx = multilabel_idx
        self.min_pred_multilabel = min_pred_multilabel
        self.is_3DLabel = is_3DLabel
        self.sampling = sampling
        self.beam_search = beam_search
        self.metric_name = metric_name
        self.set_name = set_name
        self.batch_size = batch_size
        self.each_n_epochs = each_n_epochs
        self.extra_vars = extra_vars
        self.save_path = save_path
        self.eval_on_epochs = eval_on_epochs
        self.start_eval_on_epoch = start_eval_on_epoch
        self.write_type = write_type
        self.sampling_type = sampling_type
        self.write_samples = write_samples
        self.out_pred_idx = out_pred_idx
        self.best_score = -1
        self.best_epoch = -1
        self.wait = 0
        self.verbose = verbose
        self.cum_update = 0
        self.epoch = reload_epoch
        self.max_plot = max_plot
        self.save_each_evaluation = save_each_evaluation
        self.written_header = False
        create_dir_if_not_exists(self.save_path)
        super(PrintPerformanceMetricOnEpochEndOrEachNUpdates, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        """
        On epoch end, sample and evaluate on the specified datasets.
        :param epoch: Current epoch number
        :param logs:
        :return:
        """
        epoch += 1  # start by index 1
        self.epoch = epoch
        if not self.eval_on_epochs:
            return
        if epoch < self.start_eval_on_epoch:
            if self.verbose > 0:
                logging.info('Not evaluating until end of epoch ' + str(self.start_eval_on_epoch))
            return
        elif (epoch - self.start_eval_on_epoch) % self.each_n_epochs != 0:
            if self.verbose > 0:
                logging.info('Evaluating only every ' + str(self.each_n_epochs) + ' epochs')
            return
        self.evaluate(epoch, counter_name='epoch')

    def on_batch_end(self, n_update, logs={}):
        self.cum_update += 1  # start by index 1
        if self.eval_on_epochs:
            return
        if self.cum_update % self.each_n_epochs != 0:
            return
        if self.epoch < self.start_eval_on_epoch:
            return
        self.evaluate(self.cum_update, counter_name='iteration')

    def evaluate(self, epoch, counter_name='epoch'):

        # Evaluate on each set separately
        all_metrics = []
        for s in self.set_name:
            # Apply model predictions
            if self.beam_search:
                params_prediction = {'batch_size': self.batch_size,
                                     'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                     'predict_on_sets': [s],
                                     'pos_unk': False,
                                     'heuristic': 0,
                                     'mapping': None}
                params_prediction.update(checkDefaultParamsBeamSearch(self.extra_vars))
                predictions = self.model_to_eval.predictBeamSearchNet(self.ds, params_prediction)[s]
            else:
                orig_size = self.extra_vars.get('eval_orig_size', False)
                params_prediction = {'batch_size': self.batch_size,
                                     'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                     'predict_on_sets': [s]}
                # Convert predictions
                postprocess_fun = None
                if self.is_3DLabel:
                    postprocess_fun = [self.ds.convert_3DLabels_to_bboxes, self.extra_vars[s]['references_orig_sizes']]
                elif orig_size:
                    postprocess_fun = [self.ds.resize_semantic_output, self.extra_vars[s]['eval_orig_size_id']]
                predictions = \
                    self.model_to_eval.predictNet(self.ds, params_prediction, postprocess_fun=postprocess_fun)[s]

            if self.is_text:
                if params_prediction.get('pos_unk', False):
                    samples = predictions[0]
                    alphas = predictions[1]

                    if eval('self.ds.loaded_raw_' + s + '[0]'):
                        sources = predictions[2]
                    else:
                        sources = []
                        for preds in predictions[2]:
                            for src in preds[self.input_text_id]:
                                sources.append(src)
                        sources = decode_predictions_beam_search(sources,
                                                                 self.index2word_x,
                                                                 pad_sequences=True,
                                                                 verbose=self.verbose)
                    heuristic = params_prediction['heuristic']
                else:
                    samples = predictions
                    alphas = None
                    heuristic = None
                    sources = None
                if self.out_pred_idx is not None:
                    samples = samples[self.out_pred_idx]
                # Convert predictions into sentences
                if self.beam_search:
                    predictions = decode_predictions_beam_search(samples,
                                                                 self.index2word_y,
                                                                 alphas=alphas,
                                                                 x_text=sources,
                                                                 heuristic=heuristic,
                                                                 mapping=params_prediction['mapping'],
                                                                 verbose=self.verbose)
                else:
                    predictions = decode_predictions(predictions,
                                                     1, # always set temperature to 1
                                                     self.index2word_y,
                                                     self.sampling_type,
                                                     verbose=self.verbose)

            elif self.is_multilabel:
                if self.multilabel_idx is not None:
                    predictions = predictions[self.multilabel_idx]
                predictions = decode_multilabel(predictions, 
                                                self.index2word_y, 
                                                min_val=self.min_pred_multilabel, 
                                                verbose=self.verbose)
                    
            # Store predictions
            if self.write_samples:
                # Store result
                filepath = self.save_path + '/' + s + '_' + counter_name + '_' + str(epoch) + '.pred'  # results file
                if self.write_type == 'list':
                    list2file(filepath, predictions)
                elif self.write_type == 'vqa':
                    list2vqa(filepath, predictions, self.extra_vars[s]['question_ids'])
                elif self.write_type == 'listoflists':
                    listoflists2file(filepath, predictions)
                elif self.write_type == 'numpy':
                    numpy2file(filepath, predictions)
                elif self.write_type == '3DLabels':
                    # TODO:
                    print("WRITE SAMPLES FUNCTION NOT IMPLEMENTED")
                else:
                    raise NotImplementedError(
                        'The store type "' + self.write_type + '" is not implemented.')


            # Evaluate on each metric
            for metric in self.metric_name:
                if self.verbose > 0:
                    logging.info('Evaluating on metric ' + metric)
                filepath = self.save_path + '/' + s + '.' + metric  # results file

                # Evaluate on the chosen metric
                metrics = evaluation.select[metric](
                    pred_list=predictions,
                    verbose=self.verbose,
                    extra_vars=self.extra_vars,
                    split=s)

                # Print results to file and store in model log
                with open(filepath, 'a') as f:
                    header = counter_name + ','
                    line = str(epoch) + ','
                    # Store in model log
                    self.model_to_eval.log(s, counter_name, epoch)
                    for metric_ in sorted(metrics):
                        all_metrics.append(metric_)
                        value = metrics[metric_]
                        header += metric_ + ','
                        line += str(value) + ','
                        # Store in model log
                        self.model_to_eval.log(s, metric_, value)
                    if not self.written_header:
                        f.write(header + '\n')
                        self.written_header = True
                    f.write(line + '\n')

                if self.verbose > 0:
                    logging.info('Done evaluating on metric ' + metric)

        # Plot results so far
        self.model_to_eval.plot(counter_name, set(all_metrics), self.set_name, upperbound=self.max_plot)

        # Save the model
        if self.save_each_evaluation:
            from keras_wrapper.cnn_model import saveModel
            saveModel(self.model_to_eval, epoch, store_iter=not self.eval_on_epochs)

PrintPerformanceMetricOnEpochEndOrEachNUpdates = EvalPerformance


###################################################
# Storing callbacks
###################################################
class StoreModel(KerasCallback):
    def __init__(self, model, fun, epochs_for_save, verbose=0):
        """
        In:
            model - model to save
            fun - function for saving the model
            epochs_for_save - number of epochs before the last save
        """
        super(KerasCallback, self).__init__()
        self.model_to_save = model
        self.store_function = fun
        self.epochs_for_save = epochs_for_save
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        epoch += 1
        if epoch % self.epochs_for_save == 0:
            print('')
            self.store_function(self.model_to_save, epoch)

            # def on_batch_end(self, n_update, logs={}):
            #    n_update += 1
            #    if (n_update % self.epochs_for_save == 0):
            #        print('')
            #        self.store_function(self.model_to_save, n_update)

StoreModelWeightsOnEpochEnd = StoreModel

###################################################
# Sampling callbacks
###################################################

class Sample(KerasCallback):
    def __init__(self, model, dataset, gt_id, set_name, n_samples, each_n_updates=10000, extra_vars=None,
                 is_text=False, index2word_x=None, index2word_y=None, input_text_id=None, print_sources=False,
                 sampling='max_likelihood', temperature=1.,
                 beam_search=False, batch_size=50, reload_epoch=0, start_sampling_on_epoch=0, is_3DLabel=False,
                 write_type='list', sampling_type='max_likelihood', out_pred_idx=None, in_pred_idx=None, verbose=1):
        """
            :param model: model to evaluate
            :param dataset: instance of the class Dataset in keras_wrapper.dataset
            :param gt_id: identifier in the Dataset instance of the output data about to evaluate
            :param metric_name: name of the performance metric
            :param set_name: name of the set split that will be evaluated
            :param n_samples: number of samples predicted during sampling
            :param each_n_updates: sampling each this number of epochs
            :param extra_vars: dictionary of extra variables
            :param is_text: defines if the predicted info is of type text
                            (in that case the data will be converted from values into a textual representation)
            :param is_3DLabel: defines if the predicted info is of type 3DLabels

            :param index2word_y: mapping from the indices to words (only needed if is_text==True)
            :param sampling: sampling mechanism used (only used if is_text==True)
            :param out_pred_idx: index of the output prediction used for evaluation
                            (only applicable if model has more than one output, else set to None)
            :param reload_epoch: number o the epoch reloaded (0 by default)
            :param start_sampling_on_epoch: only starts evaluating model if a given epoch has been reached
            :param in_pred_idx: index of the input prediction used for evaluation
                            (only applicable if model has more than one input, else set to None)
            :param verbose: verbosity level; by default 1
        """
        self.model_to_eval = model
        self.ds = dataset
        self.gt_id = gt_id
        self.index2word_x = index2word_x
        self.index2word_y = index2word_y
        self.input_text_id = input_text_id
        self.is_text = is_text
        self.sampling = sampling
        self.beam_search = beam_search
        self.batch_size = batch_size
        self.set_name = set_name
        self.n_samples = n_samples
        self.each_n_updates = each_n_updates
        self.is_3DLabel = is_3DLabel
        self.extra_vars = extra_vars
        self.temperature = temperature
        self.reload_epoch = reload_epoch
        self.start_sampling_on_epoch = start_sampling_on_epoch
        self.write_type = write_type
        self.sampling_type = sampling_type
        self.out_pred_idx = out_pred_idx
        self.in_pred_idx = in_pred_idx
        self.cum_update = 0
        self.epoch_count = 0
        self.print_sources = print_sources
        self.verbose = verbose
        super(SampleEachNUpdates, self).__init__()

    def on_epoch_end(self, n_epoch, logs={}):
        self.epoch_count += 1

    def on_batch_end(self, n_update, logs={}):
        self.cum_update += 1
        if self.epoch_count + self.reload_epoch < self.start_sampling_on_epoch:
            return
        elif self.cum_update % self.each_n_updates != 0:
            return

        # Evaluate on each set separately
        for s in self.set_name:
            # Apply model predictions
            params_prediction = {'batch_size': self.batch_size,
                                 'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                 'predict_on_sets': [s],
                                 'n_samples': self.n_samples,
                                 'pos_unk': False,
                                 'heuristic': 0,
                                 'mapping': None}
            if self.beam_search:
                params_prediction.update(checkDefaultParamsBeamSearch(self.extra_vars))
                predictions, truths, sources = self.model_to_eval.predictBeamSearchNet(self.ds, params_prediction)
            else:
                # Convert predictions
                postprocess_fun = None
                if self.is_3DLabel:
                    postprocess_fun = [self.ds.convert_3DLabels_to_bboxes, self.extra_vars[s]['references_orig_sizes']]
                predictions = self.model_to_eval.predictNet(self.ds, params_prediction, postprocess_fun=postprocess_fun)


            if self.print_sources:
                if self.in_pred_idx is not None:
                    sources = [srcs for srcs in sources[0][self.in_pred_idx]]
                sources = decode_predictions_beam_search(sources,
                                                         self.index2word_x,
                                                         pad_sequences=True,
                                                         verbose=self.verbose)

            if s in predictions:
                if params_prediction['pos_unk']:
                    samples = predictions[s][0]
                    alphas = predictions[s][1]
                    heuristic = params_prediction['heuristic']
                else:
                    samples = predictions[s]
                    alphas = None
                    heuristic = None

                predictions = predictions[s]
                if self.is_text:
                    if self.out_pred_idx is not None:
                        samples = samples[self.out_pred_idx]
                    # Convert predictions into sentences
                    if self.beam_search:
                        predictions = decode_predictions_beam_search(samples,
                                                                     self.index2word_y,
                                                                     alphas=alphas,
                                                                     x_text=sources,
                                                                     heuristic=heuristic,
                                                                     mapping=params_prediction['mapping'],
                                                                     verbose=self.verbose)
                    else:
                        predictions = decode_predictions(samples,
                                                         1,
                                                         self.index2word_y,
                                                         self.sampling_type,
                                                         verbose=self.verbose)
                    truths = decode_predictions_one_hot(truths, self.index2word_y, verbose=self.verbose)

                # Write samples
                if self.print_sources:
                    # Write samples
                    for i, (source, sample, truth) in enumerate(zip(sources, predictions, truths)):
                        print ("Source     (%d): %s" % (i, source))
                        print ("Hypothesis (%d): %s" % (i, sample))
                        print ("Reference  (%d): %s" % (i, truth))
                        print ("")
                else:
                    for i, (sample, truth) in enumerate(zip(predictions, truths)):
                        print ("Hypothesis (%d): %s" % (i, sample))
                        print ("Reference  (%d): %s" % (i, truth))
                        print ("")

SampleEachNUpdates = Sample

###################################################
# Learning modifiers callbacks
###################################################
class EarlyStopping(KerasCallback):
    """
    Applies early stopping if performance has not improved for some epochs.
    """

    def __init__(self,
                 model,
                 patience=0,
                 check_split='val',
                 metric_check='acc',
                 eval_on_epochs=True,
                 each_n_epochs=1,
                 start_eval_on_epoch=0,
                 verbose=1):
        """
        :param model: model to check performance
        :param patience: number of beginning epochs without reduction; by default 0 (disabled)
        :param check_split: data split used to check metric value improvement
        :param metric_check: name of the metric to check
        :param verbose: verbosity level; by default 1
        """
        super(KerasCallback, self).__init__()
        self.model_to_eval = model
        self.patience = patience
        self.check_split = check_split
        self.metric_check = metric_check
        self.eval_on_epochs = eval_on_epochs
        self.start_eval_on_epoch = start_eval_on_epoch
        self.each_n_epochs = each_n_epochs

        self.verbose = verbose
        self.cum_update = 0
        # check already stored scores in case we have loaded a pre-trained model
        all_scores = self.model_to_eval.getLog(self.check_split, self.metric_check)
        all_epochs = self.model_to_eval.getLog(self.check_split, 'epoch')
        if all_scores[-1] is not None:
            self.best_score = max(all_scores)
            self.best_epoch = all_epochs[all_scores.index(self.best_score)]
            self.wait = max(all_epochs) - self.best_epoch
        else:
            self.best_score = -1.
            self.best_epoch = -1
            self.wait = 0

    def on_epoch_end(self, epoch, logs={}):
        epoch += 1  # start by index 1
        self.epoch = epoch
        if not self.eval_on_epochs:
            return
        elif (epoch - self.start_eval_on_epoch) % self.each_n_epochs != 0:
            return
        self.evaluate(self.epoch, counter_name='epoch')

    def on_batch_end(self, n_update, logs={}):
        self.cum_update += 1  # start by index 1
        if self.eval_on_epochs:
            return
        if self.cum_update % self.each_n_epochs != 0:
            return
        self.evaluate(self.cum_update, counter_name='update')

    def evaluate(self, epoch, counter_name='epoch'):
        current_score = self.model_to_eval.getLog(self.check_split, self.metric_check)[-1]
        # Get last metric value from logs
        if current_score is None:
            warnings.warn('The chosen metric' + str(self.metric_check) + ' does not exist;'
                                                                         ' this reducer works only with a valid metric.')
            return

        # Check if the best score has been outperformed in the current epoch
        if current_score > self.best_score:
            self.best_epoch = epoch
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                logging.info('---current best %s %s: %.3f' % (self.check_split, self.metric_check, current_score))

        # Stop training if performance has not improved for self.patience epochs
        elif self.patience > 0:
            self.wait += 1
            logging.info('---bad counter: %d/%d' % (self.wait, self.patience))
            if self.wait >= self.patience:
                if self.verbose > 0:
                    logging.info("---%s %d: early stopping. Best %s found at %s %d: %f" % (
                    str(counter_name), epoch, self.metric_check,  str(counter_name), self.best_epoch, self.best_score))
                self.model.stop_training = True


class LearningRateReducer(KerasCallback):
    """
    Reduces learning rate during the training.
    """

    def __init__(self, lr_decay=1, reduce_rate=0.5, reduce_nb=99999, verbose=1):
        """
        :param lr_decay: minimum number of epochs passed before the last reduction
        :param reduce_rate: multiplicative rate reducer; by default 0.5
        :param reduce_nb: maximal number of reductions performed; by default 99999
        :param verbose: verbosity level; by default 1
        """
        super(KerasCallback, self).__init__()
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose
        self.epsilon = 0.1e-10
        self.lr_decay = lr_decay
        self.last_lr_decrease = 0

    def on_epoch_end(self, epoch, logs={}):

        # Decrease LR if self.lr_decay epochs have passed sice the last decrease
        self.last_lr_decrease += 1
        if self.last_lr_decrease >= self.lr_decay:
            self.current_reduce_nb += 1
            if self.current_reduce_nb <= self.reduce_nb:
                lr = self.model.optimizer.lr.get_value()
                self.model.optimizer.lr.set_value(np.float32(lr * self.reduce_rate))
                if self.verbose > 0:
                    logging.info("LR reduction from {0:0.6f} to {1:0.6f}". \
                                 format(float(lr), float(lr * self.reduce_rate)))
                if float(lr) <= self.epsilon:
                    if self.verbose > 0:
                        logging.info('Learning rate too small, learning stops now')
                    self.model.stop_training = True
