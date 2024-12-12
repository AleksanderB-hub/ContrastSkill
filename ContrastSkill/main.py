#Basic libraries
import argparse
import time
import random
import numpy as np
import os
import json
import logging
import math
import logging 
import copy 
import sys
import shutil  
import faiss
import torch


#For controlling the loss functions etc. 
from torch.utils.tensorboard import SummaryWriter 
#Samplers
from torch.utils.data import (DataLoader, RandomSampler, DistributedSampler)
#Importing the distributed module
import torch.distributed as dist
#Optimizer
from torch.optim import AdamW 
#from transformers import AdamW
#Getting the linear sechdule with warmup
from transformers import get_linear_schedule_with_warmup as WarmupLinearScheduler
#TQDM
from tqdm import tqdm, trange
#For torch modules
import torch.nn
#For torch functional
import torch.nn.functional as F
#for models 
from transformers import (BertConfig, BertModel, BertTokenizerFast, RobertaConfig, RobertaModel, RobertaTokenizerFast)
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import AutoModelForTokenClassification
#For testing to wokr on a sample fo data
from torch.utils.data import Subset
#For distributer training
from torch.nn.parallel import DistributedDataParallel as DDP
#For supervised evaluation 
import evaluate 
seqeval = evaluate.load('seqeval')


from backend import Info_NCE, normalize_embeddings, Contrastskill, skill_dataloader, vectorizer_sentence, BioTaggingModel, tensorize_data, predict, evaluation

#You can add other models if required as long as they are supported on hugging face
Model_Classes = {
    'bert': (BertConfig, BertModel, BertTokenizerFast),
    'roberta': (AutoConfig, AutoModel, AutoTokenizer),
    'joberta': (AutoConfig, AutoModel, AutoTokenizer)
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Training and evaluation configuration")

    #General settings
    parser.add_argument('--local_rank', type=int, default=-1, help='Rank for distributed training')
    parser.add_argument('--available_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--seed', type=int, default=21, help='Random seed for reproducibility')

    #Model settings
    parser.add_argument('--model_type', type=str, default='joberta', help='Model type')
    parser.add_argument('--model_version', type=str, default='jjzha/jobberta-base', help='Model version')
    parser.add_argument('--scheduler', type=int, default = 20, help='This is the reference training epochs number used for learning rate scheduling. It applies to the fine-tuning stage and works independently of supervised_epochs')

    #Training settings for Contrastive Pre-Training Stage
    parser.add_argument('--contrastive_train', action='store_true', help='Enable contrastive training')
    parser.add_argument('--lowercase', action='store_true', help='Convert text to lowercase (for uncased models)')
    parser.add_argument('--weight_relevant', type=float, default=1.0, help='Weight for relevant tokens')
    parser.add_argument('--training_size', type=int, default=0, help='Training size for the contrastive stage')
    parser.add_argument('--prepare_data', action='store_true', help='Prepare the Pre-training dataset (alternatively you can set it to false and provide your own data, assuming it follows the same structure)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs for the contrastive pre-training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--early_stopping_steps', type=int, default=-1, help='Exact number of training steps to be used')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epsilon_adam', type=float, default=1e-8, help='Epsilon value for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--warmup_prop', type=float, default=0.1, help='Proportion of the pre-training data to be used for the warmup')
    parser.add_argument('--accumulate_gradients', type=int, default=1, help='In case of insufficient memory gradients can be accumulated to simulate greater batch size (significantly extends the computation)')
    parser.add_argument('--gradient_threshold', type=float, default=1.0, help='A maximum threshold for gradients')
    parser.add_argument('--writer_update_steps', type=int, default=10, help='A number of steps at which the current loss, gradients and learning rate will be recorded to the scheduler')
    parser.add_argument('--dropout', type=float, default=0.1, help='A dropout proportion for the pre-training')
    parser.add_argument('--output_size', type=int, default=768, help='Size of the embedding layer')
    parser.add_argument('--limit', type=int, default=128, help='The upper bound for the sequence length, all other sequences will be padded to this value.')
    parser.add_argument('--mode', type=int, default=1, help='The training mode for the framework. 0-only pre-trains the contrastive model, 1-pre-trains the contrastive model and fine-tunes on a downstream task, 2-only fine-tunes on a downstream task (only predicts if supervised_train is set to True)')
    #Negative Sampling
    parser.add_argument('--strong_negatives', action='store_true', help='Determine whether strong negative sampling is to be used (does not work with "flip" negatives)')
    parser.add_argument('--negative_type', type=str, default='base', help='Determine the negative sampling strategy (base or flip)')
    parser.add_argument('--number_negative_pairs', type=int, default=2, help='Number of negative pairs (works only with base negative pairing type)')
    parser.add_argument('--strong_negative_prob', type=float, default=1.0, help='The probability of strong negative pair if strong__negatives is set to True')
    #Loss
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for the InfoNCE loss')
    parser.add_argument('--main_loss_weight', type=float, default=1.0, help='The weight for the InfoNCE loss (leave at default)')
    
    #Training Settings for the Fine-Tuning Stage
    parser.add_argument('--dev_experiments', action='store_true', help='This determines whether you are in the development setting or deploying a final model. Development Setting is used to determine the optimal number of training epochs for the given model dataset combination.')
    parser.add_argument('--supervised_dataset', type=str, default='SkillSpan', help='Determine which dataset you want to use [SkillSpan, Green, Sayfullina]')
    parser.add_argument('--tagging_type', type=str, default='bio', help='Sets a tagging type, default is BIO. If a different data structure were to be explored the adequate adjustments must be made to the backend.py file')
    parser.add_argument('--supervised_train', action='store_true', help='Determine whether to fine-tune the model on a downstream task')
    parser.add_argument('--supervised_raw', action='store_true', help='Determine whether to use a base model (True) or contrastive pre-trained version (False)')
    parser.add_argument('--supervised_num_labels', type=int, default=3, help='Number of labels for a downstream task. Similarly as tagging_type if other data is used, the backend must be adjusted.')
    parser.add_argument('--supervised_epochs', type=int, default=20, help='The number of epochs to be used in a supervised setting')
    parser.add_argument('--supervised_early_stopping_steps', type=int, default=-1, help='The exact number of training steps to trigger early stopping')
    parser.add_argument('--supervised_batch_size', type=int, default=16, help='Batch size for the fine-tuned model')
    parser.add_argument('--supervised_learning_rate', type=float, default=5e-5, help='Learning Rate for the fine-tuned model')
    parser.add_argument('--supervised_weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--supervised_warmup_prop', type=float, default=0.0, help='Warmup proportion for fine-tuning')
    parser.add_argument('--supervised_accumulate_gradients', type=int, default=1, help='Determine whether to simulate larger batch size and accumulate gradients')
    parser.add_argument('--dropout_sup', type=float, default=0.0, help='Dropout proportion for fine-tuned model')
    parser.add_argument('--patience', type=int, default=5, help='Patience, used for determining the optimal number of epochs for the fine-tuned model')
    parser.add_argument('--predict', action='store_true', help='If set, the fine-tuned model will be used to predict the test dataset, use only after previous experiments are performed and the best performing configuration is selected')
    parser.add_argument('--cross_dataset', type=int, default=0, help='This parameter determines whether you are testing within supervised_dataset (0) or deploying SkillSpan on Green (1) or Green on SkillSpan (2). If using 1 or 2 remember to adjust supervised_dataset parameter accordingly as it determines the training source.')
    
    #Paths
    parser.add_argument('--data_directory', type=str, required= True, help='Path to data directory, specify your path where all downstream task datasets are stored')
    parser.add_argument('--model_directory', type=str, required= True, help='Path to where your models are stored')

    args = parser.parse_args()
    return args

def set_seed(args):
    """Function to configure random seed for all sub-processes."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.available_gpus > 1:
        torch.cuda.manual_seed_all(args.seed)
        
def distributed_learning(args):
    """
    Function to configure training environment. 
    This ensures that if more than 1 GPUS are available, they will be utilised
    """
    if args.available_gpus > 1:
        args.local_rank = 0
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend= 'nccl')
    else:
        args.local_rank = -1
        args.device = torch.device(args.device)
        
def train_contrast(args, model, training_data):
    """
    This is the train function for the contrastive learning stage. It calls all of the classes defined earlier.
    It structures the training process and ensures efficient processing of data and batches. 
    In addition it also configures a summary writer so that some learning process characteristics can be investigated.
    """
    
    #Configure a SummaryWriter
    summary_writer = SummaryWriter(log_dir = (args.model_directory + '/Unsupervised/Logs'))

    #Define dataloaders and samplers
    #Determine the training batch size based on the number of GPUs
    batch_size = args.batch_size * max(1, args.available_gpus)

    #Sampler
    if args.local_rank == 0:
        train_sampler = DistributedSampler(training_data)
    else:
        train_sampler = RandomSampler(training_data)

    #Dataloader
    dataloader_training = DataLoader(training_data, sampler= train_sampler, batch_size= batch_size)

    #Determine the training parameters (epochs, weight_decay, optimizer, scheduler, gradient accumulation)

    #Account for early stopping
    #Adjust the number of gradient updates if gradient accumulation is used
    num_weight_updates = len(dataloader_training) // args.accumulate_gradients
    if args.early_stopping_steps > 0:
        #Determine the number of epochs 
        args.epochs = args.early_stopping_steps // num_weight_updates + 1
        training_steps = args.early_stopping_steps
    else:
        training_steps = num_weight_updates * args.epochs

    #Determine optimizer and scheduler 
    #Since we want to apply weight decay we need to make sure that the the embedding, normalization and bias layers are not affected

    no_weight_decay_layers = ['bias', 'LayerNorm.weight', 'embedding', 'BatchNorm.weight', 'InstanceNorm.weight', 'GroupNorm.weight']
    no_decay_params = []
    decay_params = []
    for name, params in model.named_parameters():
        if any(key in name for key in no_weight_decay_layers):
            no_decay_params.append(params)
        else:
            decay_params.append(params)

    adjusted_parameters = [
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': decay_params, 'weight_decay': args.weight_decay}
    ]
        
    #Define optimizer 
    optimizer = AdamW(params= adjusted_parameters, lr= args.learning_rate, eps= args.epsilon_adam)

    #Scheduler 
    scheduler = WarmupLinearScheduler(optimizer, num_warmup_steps= math.floor(args.warmup_prop * training_steps), num_training_steps= training_steps)
    
    #Training 
    #initialize parameters
    iterations = 0
    total_loss, previous_loss = 0.0, 0.0
    total_magnitude_accumulated, gradient_count = 0, 0

    #Update logger
    logger.info('Begin Training...')
    logger.info('Training Sample Size = %d', len(training_data))
    logger.info('Total Training Steps = %d', training_steps)
    logger.info('Epochs = %d', args.epochs)
    logger.info('Effective batch_size = %d', (args.batch_size * args.accumulate_gradients * args.available_gpus))

    #Set gradients to 0
    model.zero_grad()

    #Begin by iterating through entire training sample
    total_iterator = trange(int(args.epochs), desc='Iterating through epochs...')
    for _ in total_iterator:
        epoch_iterator = tqdm(dataloader_training, total=len(dataloader_training), desc='Iterating through batches within epoch...', mininterval= 10, miniters= 1, ncols= 100)

        #Iterate through epochs
        for step, batch in enumerate(epoch_iterator):
            model.train()
            epoch_iterator.update()
            epoch_iterator.refresh()
            #Transfer each batch into the designated device
            batch = tuple(example.to(args.device) for example in batch)
            
            #Define inputs': batch[1],
            inputs = {'anchor_ids': batch[0],
                      'anchor_input_masks': batch[1],          
                      'anchor_output_masks': batch[2],
                      'anchor_special_masks': batch[3],
                      'pairs_ids': batch[4],
                      'pairs_input_masks': batch[5],
                      'pairs_output_masks': batch[6],
                      'pairs_special_masks': batch[7],
                      'labels': batch[8]
                      }
            # print(inputs)

            #Calculate loss
            loss = model(**inputs)

            #Adjust the batch loss with the distributed training or gradient accumulation
            if args.accumulate_gradients > 1:
                loss = loss / args.accumulate_gradients

            if args.available_gpus > 1 and args.local_rank != -1:
                loss = loss.mean()

            loss.backward()

            #Clip Gradients to address exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.gradient_threshold)

            #Combine the total epoch loss (accumulate batch losses)
            total_loss += loss.item()

            #Since gradient accumulation can be performed we need to ensure that the optimizer and scheduler are updated once accumulation is finished
            if (step + 1) % args.accumulate_gradients == 0:

            #Initialize the in-batch gradient parameter (to store in-batch magnitudes)
                batch_magnitude = 0

            #Store the gradients before resetting them (for all parameters)
                for _, param in model.named_parameters():
                    if param.grad is not None: 
                        #Get the euclidean (gradient magnitude for each parm)
                        magnitude = param.grad.data.norm(2)
                        #Raise each magnitude to the power of 2
                        magnitude = magnitude ** 2
                        #Sum all the normalized gradients
                        batch_magnitude += magnitude

                #Get the overall magnitude (i.e., gradient norm for the entire batch)
                total_magnitude = batch_magnitude ** 0.5
                total_magnitude_accumulated += total_magnitude
                gradient_count += 1

                #Update the optimizer and scheduler         
                optimizer.step()
                scheduler.step()

                #Once the batch (or accumulated batch) is processed reset the gradients
                model.zero_grad()
                iterations += 1

                #Update the writer with gradients, learning rate and loss
                if iterations % args.writer_update_steps == 0:
                    average_gradient_norm = total_magnitude_accumulated / gradient_count
                    avg_loss_per_writer_update = (total_loss - previous_loss) / args.writer_update_steps
                    summary_writer.add_scalar(tag= 'gradient norms', scalar_value= average_gradient_norm, global_step= iterations)
                    summary_writer.add_scalar(tag= 'learning rate', scalar_value= scheduler.get_last_lr()[0], global_step= iterations)
                    summary_writer.add_scalar(tag= 'loss', scalar_value= avg_loss_per_writer_update, global_step= iterations)
                    previous_loss = total_loss

                    #Reset the gradients norm after each writer update
                    gradient_count = 0
                    total_magnitude_accumulated = 0  

        #Apply early stopping 
            if iterations >= args.early_stopping_steps and args.early_stopping_steps > 0:
                epoch_iterator.close() 
                total_iterator.close()
                break

        if iterations >= args.early_stopping_steps and args.early_stopping_steps > 0:
            break 

    summary_writer.close()

    #Save the model and training arguments 
    torch.save(args, os.path.join(args.model_directory, 'model_contrastive_training_args.bin'))
    if args.available_gpus > 1 and args.local_rank != -1:
        torch.save(model.module.state_dict(), os.path.join(args.model_directory, 'model_contrastive_stage.bin'))
    else:
        torch.save(model.state_dict(), os.path.join(args.model_directory, 'model_contrastive_stage.bin'))

    logger.info(f"Model was successfully trained and saved to the {args.model_directory}.")

    return print(f'Stage 1 pre-training finished overall loss is: {total_loss / iterations}')

def train_supervised(args, supervised_training_data, model, tokenizer=None, dev_data=None):
    """
    This function is used to perform a supervised training.
    The training is can be initialized with the contrastive
    pre-trained model from the stage_1. 
    Since the supervised model is simply fine-tuned transformer,
    the final model will always be a version of the initial model. 
    E.g., If the contrastive stage used BERT, supervised learning stage will utilise pre-trained BERT. 
    """
    #Configure a Logger and Summary writer 
    summary_writer = SummaryWriter(log_dir = (args.model_directory + '/Supervised/Logs'))

    #Define dataloaders and samplers
    #Determine the training batch size based on the number of GPUs
    batch_size = args.supervised_batch_size * max(1, args.available_gpus)

    #Sampler
    if args.local_rank == 0:
        train_sampler = DistributedSampler(supervised_training_data)
    else:
        train_sampler = RandomSampler(supervised_training_data)

    data_loader_supervised = DataLoader(supervised_training_data, sampler= train_sampler, batch_size= batch_size)
    
    #Account for early stopping and gradient accumulation 
    #Adjust the number of gradient updates if gradient accumulation is used
    num_weight_updates = len(data_loader_supervised) // args.supervised_accumulate_gradients
    if args.supervised_early_stopping_steps > 0:
        #Determine the number of epochs 
        args.supervised_epochs = args.supervised_early_stopping_steps // num_weight_updates + 1
        training_steps = args.supervised_early_stopping_steps
    else:
        training_steps = num_weight_updates * args.scheduler
    
    #Weight Decay
    no_weight_decay_layers = ['bias', 'LayerNorm.weight', 'embedding', 'BatchNorm.weight', 'InstanceNorm.weight', 'GroupNorm.weight']
    no_decay_params = []
    decay_params = []
    for name, params in model.named_parameters():
        if any(key in name for key in no_weight_decay_layers):
            no_decay_params.append(params)
        else:
            decay_params.append(params)

    adjusted_parameters = [
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': decay_params, 'weight_decay': args.weight_decay}
    ]
    
    optimizer = AdamW(adjusted_parameters, eps= args.epsilon_adam, lr= args.supervised_learning_rate)
    scheduler = WarmupLinearScheduler(optimizer=optimizer, num_warmup_steps= math.floor(training_steps * args.supervised_warmup_prop), num_training_steps= training_steps) 
    
    #Begin training 
    #initialize parameters
    iterations = 0
    total_loss, previous_loss = 0.0, 0.0
    total_magnitude_accumulated, gradient_count = 0, 0
    previous_best_f1, patience = 0, 0

    #Update logger
    logger.info('Begin Training...')
    logger.info('Training Sample Size = %d', len(supervised_training_data))
    logger.info('Total Training Steps (scheduler) = %d', training_steps)
    logger.info('Reference Epochs (scheduler) = %d', args.scheduler)
    logger.info('Actual Training Epochs = %d', args.supervised_epochs)
    logger.info('Effective batch_size = %d', args.supervised_batch_size * args.supervised_accumulate_gradients * (args.available_gpus))
    
    model.zero_grad()
    
    total_iterator = trange(int(args.supervised_epochs), desc= 'Iterating through epochs...')
    for epoch in total_iterator:
        epoch_iterator = tqdm(data_loader_supervised, total=len(data_loader_supervised), desc='Iterating through batches within epoch...', mininterval= 10, miniters= 1, ncols= 100)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            epoch_iterator.update()
            epoch_iterator.refresh()
            batch = tuple(example.to(args.device) for example in batch)
            inputs = {
                'inputs_ids': batch[0],
                'inputs_mask': batch[1],
                'labels_mask': batch[2]
            }
            
            loss = model(**inputs)
            
            #Adjust for the gradient accumulation and distributed environments
            if args.supervised_accumulate_gradients > 1:
                loss = loss / args.supervised_accumulate_gradients
                
            if args.available_gpus > 1 and args.local_rank != -1:
                loss = loss.mean()
                
            loss.backward()
            
            #Clip Gradients to address exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.gradient_threshold)
            
            #Accumulate the total loss so far
            total_loss += loss.item()
            if (step + 1) % args.supervised_accumulate_gradients == 0:
                
                #Initialize the in-batch gradient parameter (to store in-batch magnitudes)
                batch_magnitude = 0

                #Store the gradients before resetting them (for all parameters)
                for _, param in model.named_parameters():
                    if param.grad is not None: 
                        #Get the euclidean (gradient magnitude for each parm)
                        magnitude = param.grad.data.norm(2)
                        #Raise each magnitude to the power of 2
                        magnitude = magnitude ** 2
                        #Sum all the normalized gradients
                        batch_magnitude += magnitude

                #Get the overall magnitude (i.e., gradient norm for the entire batch)
                total_magnitude = batch_magnitude ** 0.5
                total_magnitude_accumulated += total_magnitude
                gradient_count += 1
                
            #update the optimizer and scheduler
            optimizer.step()
            scheduler.step()
            
            #Reset the gradients
            model.zero_grad()
            iterations += 1
            
            #Update to a Summary Writer
            
            if iterations % args.writer_update_steps == 0:
                gradient_norm = total_magnitude_accumulated / gradient_count
                avg_loss_per_update = (total_loss - previous_loss) / args.writer_update_steps
                summary_writer.add_scalar(tag= 'gradient_norm', scalar_value= gradient_norm, global_step= iterations)
                summary_writer.add_scalar(tag= 'learning_rate', scalar_value= scheduler.get_last_lr()[0], global_step= iterations)
                summary_writer.add_scalar(tag= 'loss', scalar_value= avg_loss_per_update, global_step= iterations)
                previous_loss = total_loss 
                
            if iterations >= args.supervised_early_stopping_steps and (args.supervised_early_stopping_steps > 0 or patience >= args.patience):
                #scheduler.close()
                break
        if dev_data:

            print('Making prediction for current epoch...')
            f1_step, precision_step, recall_step = evaluation(args, dev_data, model, tokenizer, final=False)

            if f1_step > previous_best_f1:
                print(f'The F1 score improved in epoch {epoch + 1}. New F1: {f1_step}, Precision: {precision_step} and Recall: {recall_step}, Previous best F1: {previous_best_f1}')
                patience = 0 
                previous_best_f1 = f1_step


                #Delete the previous model (if it exists)
                if os.path.exists(os.path.join(args.model_directory, 'model_supervised_stage.bin')):
                    os.remove(os.path.join(args.model_directory, 'model_supervised_stage.bin'))
                if os.path.exists(os.path.join(args.model_directory, 'model_supervised_training_args.bin')):
                    os.remove(os.path.join(args.model_directory, 'model_supervised_training_args.bin'))

                #Save the model 
                torch.save(args, os.path.join(args.model_directory, 'model_supervised_training_args.bin'))
                if args.available_gpus > 1 and args.local_rank != -1:
                    torch.save(model.module.state_dict(), os.path.join(args.model_directory, 'model_supervised_stage.bin'))
                else:
                    torch.save(model.state_dict(), os.path.join(args.model_directory, 'model_supervised_stage.bin'))
            else:
                print(f'F1 score did not improve in epoch {epoch + 1}. Current F1: {f1_step}, Precision: {precision_step} and Recall: {recall_step}, Best F1: {previous_best_f1}')
                patience += 1
                
        if iterations >= args.supervised_early_stopping_steps and (args.supervised_early_stopping_steps > 0 or patience >= args.patience):
            print(f'The F1 score has not improved over consecutive {patience} runs, or the early steps defined was reached')
            #scheduler.close()
            break
        
    summary_writer.close()
    
    #Saving the final model 
    if dev_data is None:
        torch.save(args, os.path.join(args.model_directory, 'model_supervised_training_args.bin'))
        if args.available_gpus > 1 and args.local_rank != -1:
            torch.save(model.module.state_dict(), os.path.join(args.model_directory, 'model_supervised_stage.bin'))
        else:
            torch.save(model.state_dict(), os.path.join(args.model_directory, 'model_supervised_stage.bin'))
        

    logger.info(f"Model was successfully trained and saved to the {args.model_directory}.")

    return print(f'Stage 2 fine-tunning finished overall loss is: {total_loss / iterations}')
def model_deploy(args):
    "This function is used to deploy the entire model architecture"

    if args.mode == 0 and not args.predict:
        if os.path.exists(os.path.join(args.model_directory, 'model_contrastive_stage.bin')):
            response = input(f"Contrastive Stage Pre-training: Output directory ({os.path.join(args.model_directory, 'model_contrastive_stage.bin')}) already exists and is not empty. Do you want to overwrite? (Y/N): ")
            if response.lower() in ['yes', 'y']:
                print('Only the Stage_1 (Contrastive pre-training) will be performed')
            else:
                print('Model already pre-trained.')
                sys.exit()

    elif args.mode == 1 and not args.predict:
        if os.path.exists(os.path.join(args.model_directory, 'model_supervised_stage.bin')) and os.path.exists(os.path.join(args.model_directory, 'model_contrastive_stage.bin')):
            response = input(f"Output directory for pre-trained model: ({os.path.join(args.model_directory, 'model_contrastive_stage.bin')}) and fine-tuned model ({os.path.join(args.model_directory, 'model_supervised_stage.bin')}) already exists and is not empty. Do you want to overwrite? (Y/N): ")
            if response.lower() in ['yes', 'y']:
                print('Both Stages of the pipeline will be trained')
            else:
                print('Both models are ready.')
                sys.exit()

    elif args.mode == 2 and not args.predict:
        if os.path.exists(os.path.join(args.model_directory, 'model_supervised_stage.bin')):
            response = input(f"Supervised Fine Tuning Stage: Output directory ({os.path.join(args.model_directory, 'model_supervised_stage.bin')}) already exists and is not empty. Do you want to overwrite? (Y/N): ")
            if response.lower() in ['yes', 'y']:
                print('Only the Stage_2 (Supervised fine-tuning) will be performed')
            else:
                print('Model already fine-tuned.')
                sys.exit()
                
    elif args.mode not in [0, 1, 2]:
        print('The training mode specified incorrectly. Refer to the "mode" argument.')
        sys.exit()

    if args.mode in [0, 1] and args.contrastive_train:
        #Configure logger 
        logger = logging.getLogger(__name__)
        #Configure logging 
        FORMAT = '%(asctime)s - %(levelname)s -  %(module)s - %(funcName)s - %(message)s'
        DATEFORMAT = '%d/%m/%Y %H:%M:%S:'
        logging.basicConfig(level= logging.INFO, format= FORMAT, datefmt= DATEFORMAT)
        logging.info(f"The training begins for mode {args.mode}, the local rank is {args.local_rank}, Device: {args.device}")

        #Set the training seed
        # random.seed(21)
        # np.random.seed(21)
        # torch.manual_seed(21)
        # if args.get('available_gpus') > 1:
        #     torch.cuda.manual_seed_all(21)
        set_seed(args)
    
        #Configuration for distributed learning
        distributed_learning(args)

        #Apply synchronization to all subprocesses unless they are master process [0] or non-distributed learning [-1]
        #This ensures only a master process will load the model
        if args.local_rank not in [-1, 0]:
            dist.barrier()

        #Initialize the base model for training
        if 'uncased' in args.model_version and not args.lowercase:
            #print('The uncased base model is used and sequences are Cased, training cannot continue')
            logger.warning(f"The uncased base model is used and sequences are Cased, training cannot continue, consider changing 'lowercase' or 'model_version' parameter.")
            sys.exit()
        config_class, base_model_class, tokenizer_class = Model_Classes[args.model_type]
        config = config_class.from_pretrained(args.model_version)
        base_model = base_model_class.from_pretrained(args.model_version, config= config).to(args.device)
        if args.model_type.lower() in ['roberta', 'joberta']:
            tokenizer = tokenizer_class.from_pretrained(args.model_version, add_prefix_space=True, use_fast = True)
        else: 
            tokenizer = tokenizer_class.from_pretrained(args.model_version, do_lower_case= args.lowercase)
            
        model_contrast = Contrastskill(args, config, base_model)

        #The continuation of the synchronization, sub-processes are halted until the master process loads the model,.
        if args.local_rank == 0:
            dist.barrier()

        model_contrast.to(args.device)

        #In case of the distributed training load the model as data DistributedDataParallel torch class
        if args.available_gpus > 1:
            model_contrast = DDP(model_contrast)

        #Load the dataloader
        dataloader = skill_dataloader(args, tokenizer, model_contrast)

        #When training data was not prepared before
        if args.prepare_data:
            
            print('Pairs are being formed for training...')
            pre_data_path = os.path.join(args.data_directory, 'Pre-training')
            with open(pre_data_path + '/selected_positives.json', 'r') as f:
                positives = json.load(f)
            with open(pre_data_path + '/selected_negatives.json', 'r') as f:
                negatives = json.load(f)
            training_dataset = dataloader.pair_data(positive_examples= positives, negative_examples= negatives)

        elif not args.prepare_data:
            prepared_data_path = os.path.join(args.data_directory, 'Prepared')
            print('Training Data ready...')
            with open(prepared_data_path + '/training_dataset.json', 'r') as f:
                training_dataset = json.load(f)

        train_contrast(args, model= model_contrast, training_data= training_dataset)
    
    if args.mode in [1, 2] and args.supervised_train:
        #Configure logger
        logger = logging.getLogger(__name__)
        #Configure logging 
        FORMAT = '%(asctime)s - %(levelname)s -  %(module)s - %(funcName)s - %(message)s'
        DATEFORMAT = '%d/%m/%Y %H:%M:%S:'
        logging.basicConfig(level= logging.INFO, format= FORMAT, datefmt= DATEFORMAT)
        logging.info(f"The training begins for mode {args.mode}, the local rank is {args.local_rank}, Device: {args.device}")
        
        set_seed(args)
        distributed_learning(args)
        
        if args.local_rank not in [-1, 0]:
            dist.barrier()

        #Initialize the model for training 
        if 'uncased' in args.model_version and not args.lowercase:
            #print('The uncased base model is used and sequences are Cased, training cannot continue')
            logger.warning(f"The uncased base model is used and sequences are Cased, training cannot continue, consider changing 'lowercase' or 'model_version' parameter.")
            sys.exit()
            
        config_class, base_model_class, tokenizer_class = Model_Classes[args.model_type]
        config = config_class.from_pretrained(args.model_version)
        base_model = base_model_class.from_pretrained(args.model_version, config= config).to(args.device)
        if args.model_type.lower() in ['roberta', 'joberta']:
            tokenizer = tokenizer_class.from_pretrained(args.model_version, add_prefix_space=True, use_fast = True)
        else: 
            tokenizer = tokenizer_class.from_pretrained(args.model_version, do_lower_case= args.lowercase)

            
        if not args.supervised_raw:
            print('The pre-trained model from contrastive_stage is used for supervised training')
            #Load the pre-trained model
            state_dict = torch.load(f"{args.model_directory}/model_contrastive_stage.bin")
            #Replace the layer names to match that of the base_encoder
            updated_state_dict = {k.replace('base_encoder.', ''): v for k, v in state_dict.items() if k.startswith('base_encoder.')}
            #Update the weights of the base model
            base_model.load_state_dict(updated_state_dict, strict=False)
            #Configure the supervised model
            supervised_model = BioTaggingModel(args, model= base_model, num_labels = args.supervised_num_labels)
        else:
            print('The base fine-tuned model is used for supervised training')
            supervised_model = BioTaggingModel(args, model= base_model, num_labels = args.supervised_num_labels)
            
        if args.local_rank == 0:
            dist.barrier()
            
        supervised_model.to(args.device)
        
        if args.available_gpus > 1:
            supervised_model = DDP(supervised_model)
            
        #prepare the training data. In this case there is no need for a designated dataloader since tensorize_data function handles all pre-processing
        supervised_training_data = []
        sup_data_path = os.path.join(args.data_directory, 'Supervised', args.supervised_dataset)
        #For finding optimal epoch number for full testing
        if args.dev_experiments:
            dev_data = []
            with open(sup_data_path+'/train.json', 'r') as f:
                for line in f:
                    supervised_training_data.append(json.loads(line.strip()))
            with open(sup_data_path+'/dev.json', 'r') as f:
                for line in f:
                    dev_data.append(json.loads(line.strip()))
                    
            supervised_training_dataset = tensorize_data(args, supervised_training_data, tokenizer)
            
            if args.supervised_train:
                train_supervised(args, supervised_training_dataset, supervised_model, tokenizer, dev_data)
                
        #For training the final model
        else:
            test_data = []
            with open(sup_data_path+'/total_train.json', 'r') as f:
                for line in f:
                    supervised_training_data.append(json.loads(line.strip()))
            
            #Configurations for cross-dataset scenarios
            if args.cross_dataset == 1:
                test_data_path = os.path.join(args.data_directory, 'Supervised/Green')
            elif args.cross_dataset == 2:
                test_data_path = os.path.join(args.data_directory, 'Supervised/SkillSpan')
            else:
                test_data_path = os.path.join(args.data_directory, 'Supervised', args.supervised_dataset)
                
            with open(test_data_path+'/test.json', 'r') as f:
                for line in f:
                    test_data.append(json.loads(line.strip()))
                    
            supervised_training_dataset = tensorize_data(args, supervised_training_data, tokenizer)
            
            if args.supervised_train:
                train_supervised(args, supervised_training_dataset, supervised_model, tokenizer, test_data)
        
    if args.predict:
        set_seed(args)
        print('Loading the fine tuned model for prediction')
        config_class, base_model_class, tokenizer_class = Model_Classes[args.model_type]
        config = config_class.from_pretrained(args.model_version)
        base_model = base_model_class.from_pretrained(args.model_version, config= config).to(args.device)
        if args.model_type.lower() in ['roberta', 'joberta']:
            tokenizer = tokenizer_class.from_pretrained(args.model_version, add_prefix_space=True, use_fast = True)
        else: 
            tokenizer = tokenizer_class.from_pretrained(args.model_version, do_lower_case= args.lowercase)
        #Initialize the BIO model
        supervised_model = BioTaggingModel(args, model= base_model, num_labels = args.supervised_num_labels)
        try:
            supervised_model.load_state_dict(torch.load(f"{args.model_directory}/model_supervised_stage.bin"))
            supervised_model.to(args.device)
            print('model weights loaded')
        except AttributeError:
            supervised_model = torch.nn.DataParallel(supervised_model) 
            supervised_model.load_state_dict(torch.load(f"{args.model_directory}/model_supervised_stage.bin"))
            supervised_model = supervised_model.module
            if args.available_gpus > 1 and args.local_rank != -1:
                supervised_model = DDP(supervised_model)
            supervised_model.to(args.device)
        #Load the test data
        supervised_test_data = []
        #Determine the test set according to the cross_dataset parameter
        if args.cross_dataset == 1:
            test_data_path = os.path.join(args.data_directory, 'Supervised/Green/test.json')
        elif args.cross_dataset == 2:
            test_data_path = os.path.join(args.data_directory, 'Supervised/SkillSpan/test.json')
        else:
            test_data_path = os.path.join(args.data_directory, 'Supervised', args.supervised_dataset, 'test.json')
            
        with open(test_data_path, 'r') as f:
            for line in f:
                supervised_test_data.append(json.loads(line.strip()))
        #Predict and Evaluate 
        evaluation(args, supervised_test_data, supervised_model, tokenizer, final=True)
        torch.cuda.empty_cache()
        print('All models trained, see evaluation above')
        
if __name__ == "__main__":
    #Define the logger
    logger = logging.getLogger(__name__)
    
    #Parse parameters
    args = parse_args()
    
    #set device
    device = args.device
    
    #Run the model
    model_deploy(args)