import logging
import numpy as np
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import torch
import random
import json
import torch.distributed as dist
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import copy
import faiss
import evaluate
seqeval = evaluate.load('seqeval')

class Info_NCE(torch.nn.Module):
    """
    This loss is the implementation of the InfoNCE loss
    """
    def __init__(self, args):

        super().__init__()

        self.args = args
        self.temperature = args.temperature
        
    def forward(self, logits, labels):

        labels = labels.float()

        #Scale by temperature 
        logits /= self.temperature

        #Apply cross entropy
        loss = F.cross_entropy(logits, labels, reduction='mean')

        return loss 
    
class margin_loss_border(torch.nn.Module):
    """This loss can be used in conjunction with InfoNCE loss. The goal here is to introduce a 
    more definite cut-off point for positive definitions, such as the loss will force for the positve 
    pairs to be close together and """
    def __init__(self, args):
        super(margin_loss_border, self).__init__()
        self.args = args
        self.margin = self.args.positive_margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
    
    def forward(self, inputs, labels):
        
        #the binary labels are used to retrieve the positive example
        positive_logits = inputs[labels == 1].unsqueeze(1)
        #negative_logits = inputs[labels == 0].unsqueeze(1)

        #Repeat positive logits to match the number of negatives per example for direct comparison
        #repeated_positive_logits = positive_logits.repeat(1, inputs.size(1) - 1)
        repeated_positive_logits = positive_logits.repeat_interleave(inputs.size(1) - 1, dim=1)
        
        #print(f"The repeated logits shape is: {repeated_positive_logits.shape}")
        #Zero tensor with the shape of negative pairs
        baseline = torch.zeros_like(repeated_positive_logits)
        
        #print(f"The baseline logits shape is: {baseline.shape}")
        #Create a tensor of ones for the targets, indicating positives should be ranked higher
        targets = torch.ones_like(repeated_positive_logits)
        #print(f"The targets logits shape is: {targets.shape}")

        #Calculate loss: enforce positive logits are higher than zero by at least the margin
        loss = self.ranking_loss(repeated_positive_logits, baseline, targets)

        return loss
    
def normalize_embeddings(embeddings):
    """
    Normalize the embeddings using L2 norm.
    """ 
    sentence_embeddings = []
    for key in embeddings:
        embedding = np.asarray(embeddings[key])
        norm = np.linalg.norm(embedding)
        if norm <= 0:
            print('incorrect_sentence')
        normalized_emb = embedding / (norm if norm > 0 else 1e-9)
        sentence_embeddings.append(normalized_emb)
    final_embeddings = np.asarray(sentence_embeddings).astype(np.float32)

    return final_embeddings   

#Class for the model
class Contrastskill(torch.nn.Module):
    def __init__(self, args, config= None, base_encoder= None):

        super().__init__()
        #General attributes
        self.args = args
        self.device = self.args.device
        self.main_loss_weight = self.args.main_loss_weight
        #This is solely for testing since the deployment pipeline loads up the config and models on its own 
        if config is not None:
            self.config = config
        #This is solely for testing since the deployment pipeline loads up the config and models on its own 
        if base_encoder is not None:
            self.base_encoder = base_encoder
        
        #Define other attributes
        self.negative_aggregator = nn.Linear(self.config.hidden_size, self.args.output_size)
        self.dropout = nn.Dropout(self.args.dropout)
        self.cosine = nn.CosineSimilarity(dim= 2)
        self.weight_relevant = args.weight_relevant
        self.main_loss = Info_NCE(args)
        if self.main_loss_weight < 1.0:
            self.margin_loss = margin_loss_border(args)
            
    #forward function for learning
    def forward(self, anchor_ids= None, anchor_input_masks= None, anchor_output_masks= None, anchor_special_masks= None, pairs_ids= None, pairs_input_masks= None, pairs_output_masks= None, pairs_special_masks= None, labels= None):
        """
        This function is used to inform signals in the contrastive model.
        It takes the prepared pairs (either weak or strong) and learns the CL 
        objective based on the loss function.
        """
        #For initial embedding generation 
        if labels is None:
            if anchor_ids is not None:
                return self.encode(anchor_ids, anchor_input_masks, anchor_output_masks, anchor_special_masks)
            
        batch_size = pairs_ids.shape[0]
        number_pairs = pairs_ids.shape[1]

        #Vectorize anchors
        anchor_embeddings = self.encode(anchor_ids, anchor_input_masks, anchor_output_masks, anchor_special_masks)
        #These needs to be expanded to match the dimension of pair embeddings to compare similarity
        anchor_embeddings = anchor_embeddings.unsqueeze(1).expand(-1, number_pairs, -1)

        #Create masks for positive and negative pairs
        positive_mask = labels == 1
        negative_mask = labels == 0

        #positive pairs (no need to reshape since there is only one positive pair)
        positive_pairs_ids = pairs_ids[positive_mask]
        positive_pairs_input_masks = pairs_input_masks[positive_mask]
        positive_pairs_output_masks = pairs_output_masks[positive_mask]
        positive_pairs_special_masks = pairs_special_masks[positive_mask]

        #negative pairs (need to reshape from [batch_size, num_pairs, sentence_length] ---> [batch_size * num_pairs, sentence_length] to ensure parallel processing)
        negative_pairs_ids = pairs_ids[negative_mask].view(batch_size * (number_pairs - 1), -1)
        negative_pairs_input_masks = pairs_input_masks[negative_mask].view(batch_size * (number_pairs - 1), -1)
        negative_pairs_output_masks = pairs_output_masks[negative_mask].view(batch_size * (number_pairs - 1), -1)
        negative_pairs_special_masks = pairs_special_masks[negative_mask].view(batch_size * (number_pairs - 1), -1)

        #encode both
        positive_embeddings = self.encode(positive_pairs_ids, positive_pairs_input_masks, positive_pairs_output_masks, positive_pairs_special_masks)
        #add a new [pair] dimension to ensure proper concatenation in this case there is only one positive pair hence '1'
        positive_embeddings = positive_embeddings.unsqueeze(1)

        negative_embeddings = self.encode(negative_pairs_ids, negative_pairs_input_masks, negative_pairs_output_masks, negative_pairs_special_masks)
        negative_embeddings = negative_embeddings.view(batch_size, (number_pairs - 1), negative_embeddings.size(-1))

        #Combine both positives and negatives for logits computation 
        pair_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=1)

        pair_embeddings = pair_embeddings.view(batch_size, number_pairs, -1)

        #Compute the logits (cosine-sim) between each pair
        logits = self.cosine(anchor_embeddings, pair_embeddings)

        #Apply the loss function
        if self.main_loss_weight < 1.0:
            main_loss = self.main_loss(logits, labels)
            margin_loss = self.margin_loss(logits, labels)
            loss = self.main_loss_weight * main_loss + (1 - self.main_loss_weight) * margin_loss
        else:
            loss = self.main_loss(logits, labels)
            
        return loss
    
        

    def encode(self, input_ids= None, input_mask= None, output_mask= None, special_mask= None, drop_out= True, function= None):
        
        batch_size = input_ids.shape[0]
        weight_relevant = self.weight_relevant
        
        base_vectors = self.base_encoder(input_ids, attention_mask = input_mask)[0] #last_hidden_state

        #Ensuring the shapes of the embeddings tensors are consistent 
        base_vectors = base_vectors.view(batch_size, input_ids.shape[-1], self.config.hidden_size)

        #Creating a relevant mask (exclude all special tokens in weighted approach)
        weighted_mask = output_mask * weight_relevant + (1-output_mask) * (1 - special_mask) * (1 - weight_relevant) 
        
        #Getting the representation
        final_vectors = base_vectors * weighted_mask.unsqueeze(-1)

        #Applying the dropout
        if drop_out:
            final_vectors = self.dropout(final_vectors)

        #aggregate the vectors
        aggregated_vectors = torch.sum(final_vectors, dim=1)

        #normalize the sum of weights to avoid overly large values for multiple relevant tokens or longer sequences
        sum_of_weights = torch.sum(weighted_mask, dim=1, keepdim=True)
        sum_of_weights = torch.clamp(sum_of_weights, min=1e-9)

        final_representation = aggregated_vectors / sum_of_weights

        return final_representation
            
    def vectorizer_sentence(self, sentences, dataloader, batch_size_exact = 64):

        sentence_ids, sentence_input_masks, sentence_output_masks, sentence_special_masks, ids = [], [], [], [], []

        for i, sentence in enumerate(sentences):
            example = sentence['Tokens']
            rel_mask = sentence['relevant_mask'] 
            
            input_ids, input_mask, output_mask, special_mask, _ = dataloader.tensorize_sentence(example, rel_mask)
            sentence_ids.append(input_ids)
            sentence_input_masks.append(input_mask)
            sentence_output_masks.append(output_mask)
            sentence_special_masks.append(special_mask)
            ids.append(i)

        #Convert to tensors
        ids = torch.tensor(ids, dtype= torch.long)
        sentence_ids = torch.tensor(sentence_ids, dtype= torch.long)
        sentence_input_masks = torch.tensor(sentence_input_masks, dtype= torch.long)
        sentence_output_masks = torch.tensor(sentence_output_masks, dtype= torch.long)
        sentence_special_masks = torch.tensor(sentence_special_masks, dtype= torch.long)

        #Convert to Torch dataset
        dataset = TensorDataset(ids, sentence_ids, sentence_input_masks, sentence_output_masks, sentence_special_masks)
        
        #Configuring for multiGPU 
        batch_size = batch_size_exact * max(1, self.args.available_gpus)

        #Define a sampler 
        sentence_sampler = SequentialSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset)

        #Importing dataloader
        dataloader = DataLoader(dataset, batch_size= batch_size, sampler= sentence_sampler)

        #create representations 
        sentences_embeddings = {}
        for batch in tqdm(dataloader, desc= 'Vectorizing_sentences...'):
            self.eval()
            #Transfer batches to GPU if possible 
            batch = tuple(tensor_sentence.to(self.args.device) for tensor_sentence in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[1],
                          'input_mask': batch[2],
                          'output_mask': batch[3],
                          'special_mask': batch[4]}
                ids = batch[0].tolist()
                sentence_representation = self.encode(**inputs)
                sentence_representation = sentence_representation.tolist()
                batch_embeddings = {sentence_id: sentence_representation[i] for i, sentence_id in enumerate(ids)}
                sentences_embeddings.update(batch_embeddings)

        return sentences_embeddings
    
def vectorizer_sentence(args, sentences, dataloader, model, batch_size_exact=64):
    """
    Unlike the previous vectorizer function, this calls an already loaded model to embed the sequences. It is used in Training
    """

    sentence_ids, sentence_input_masks, sentence_output_masks, sentence_special_masks, ids = [], [], [], [], []

    for i, sentence in enumerate(sentences):
        example = sentence['Tokens']
        rel_mask = sentence['relevant_mask'] 
            
        input_ids, input_mask, output_mask, special_mask, _ = dataloader.tensorize_sentence(example, rel_mask)
        sentence_ids.append(input_ids)
        sentence_input_masks.append(input_mask)
        sentence_output_masks.append(output_mask)
        sentence_special_masks.append(special_mask)
        ids.append(i)

    #Convert to tensors
    ids = torch.tensor(ids, dtype= torch.long)
    sentence_ids = torch.tensor(sentence_ids, dtype= torch.long)
    sentence_input_masks = torch.tensor(sentence_input_masks, dtype= torch.long)
    sentence_output_masks = torch.tensor(sentence_output_masks, dtype= torch.long)
    sentence_special_masks = torch.tensor(sentence_special_masks, dtype= torch.long)

    #Convert to Torch dataset
    dataset = TensorDataset(ids, sentence_ids, sentence_input_masks, sentence_output_masks, sentence_special_masks)
    
    #Configuring for multiGPU 
    batch_size = batch_size_exact * max(1, args.available_gpus)

    #Define a sampler 
    sentence_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)

    #Importing dataloader
    dataloader = DataLoader(dataset, batch_size= batch_size, sampler= sentence_sampler)

    #create representations 
    sentences_embeddings = {}
    for batch in tqdm(dataloader, desc= 'Vectorizing_sentences...'):
        #Transfer batches to GPU if possible 
        batch = tuple(tensor_sentence.to(args.device) for tensor_sentence in batch)
        with torch.no_grad():
            inputs = {'anchor_ids': batch[1],
                      'anchor_input_masks': batch[2],
                      'anchor_output_masks': batch[3],
                      'anchor_special_masks': batch[4]}
            ids = batch[0].tolist()
            sentence_representation = model(**inputs)
            sentence_representation = sentence_representation.tolist()
            batch_embeddings = {sentence_id: sentence_representation[i] for i, sentence_id in enumerate(ids)}
            sentences_embeddings.update(batch_embeddings)

    return sentences_embeddings


#Class for dataloader
class skill_dataloader:
    def __init__(self, args, tokenizer, model=None):
        self.args = args
        self.tokenizer = tokenizer
        self.device = self.args.device
        if self.args.model_type == 'bert':
            self.cls_token = '[CLS]'
            self.sep_token = '[SEP]'
        else:
            self.cls_token = '<s>'
            self.sep_token = '</s'

        #Get the args
        self.strong_negatives = self.args.strong_negatives
        self.seed = self.args.seed
        self.number_negative_pairs = self.args.number_negative_pairs
        self.strong_negative_prob = self.args.strong_negative_prob

        self.tensorized_negatives = list()
        self.tensotized_positives = list()

        self.stage = self.args.mode
        if self.stage in [0, 1]:
            self.vectorizer = model
        self.limit_sentence = self.args.limit
        self.total_neg_padded = 0
        self.training_size = args.training_size
        self.negative_type = args.negative_type
  
    def tensorize_sentence(self, tokens, rel_positions=None):
        """
        This function tensorizes the sequences in respect to their relevance masks.
        """

        tensor_input_ids = [self.tokenizer.convert_tokens_to_ids(self.cls_token)]
        tensor_input_mask = [1]
        #[CLS] token is not "relevant" since we do not classify in this part
        tensor_output_mask = [0]
        tensor_special_mask = [1]
        word_tokens = []

        #the same upper limit is applied to the negative examples
        limit = self.limit_sentence

        #Convert tokens to corresponding tensor masks 
        for idx, token in enumerate(tokens):
            token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(token))
            #Track new tokens in respect to potential sub-tokens generation
            word_tokens.append((len(tensor_input_ids), len(tensor_input_ids) + len(token_id), token_id))
            tensor_input_ids += token_id
            tensor_input_mask += [1] * len(token_id)
            tensor_special_mask += [0] * len(token_id)
            #Different for positive and negative examples (all tokens are relevant in negatives)
            if rel_positions == None:
                tensor_output_mask += [1] * len(token_id)
            else:
                tensor_output_mask += [rel_positions[idx]] * len(token_id)

        #Shorten then sequences (only applies to the negatives as they are longer)
        if len(tensor_input_ids) > (limit -1):
            tensor_input_ids = tensor_input_ids[:limit -1]
            tensor_input_mask = tensor_input_mask[:limit -1]
            tensor_output_mask = tensor_output_mask[:limit -1]
            tensor_special_mask = tensor_special_mask[:limit -1]
            self.total_neg_padded += 1

        #[SEP] tokens
        tensor_input_ids += [self.tokenizer.convert_tokens_to_ids(self.sep_token)]
        tensor_input_mask += [1]
        tensor_output_mask += [0]
        tensor_special_mask += [1]

        #Append all final lists and pad the sequences
        padding_len = limit - len(tensor_input_ids)
        tensor_input_ids += [0] * padding_len
        tensor_input_mask += [0] * padding_len
        tensor_output_mask += [0] * padding_len
        tensor_special_mask += [1] * padding_len 

        assert len(tensor_input_ids) == limit
        assert len(tensor_input_mask) == limit
        assert len(tensor_output_mask) == limit
        assert len(tensor_special_mask) == limit

        return tensor_input_ids, tensor_input_mask, tensor_output_mask, tensor_special_mask, word_tokens
    
    def form_pairs(self, positive_examples, max_examples_per_competence=40, min_examples=5, remove_insufficient=True):
        """
        This function creates positive pairs of the data with balanced distribution.
        Pairs are formed based on the competence labels, ensuring a controlled number of pairs per competence.
        
             :param positive_examples: A list of examples containing competences.
        :param max_examples_per_competence: Maximum number of examples per competence.
        :param min_examples: Minimum number of examples needed for pairing.
        :return: A list of positive pairs and a dictionary of competences with insufficient examples and required count.
        """
        #Define seed
        random.seed(self.args.seed)
        
        #Group by the competence label
        competence_groups = {}
        for example in positive_examples:
            competence = example['competence']
            if competence not in competence_groups:
                competence_groups[competence] = []
            competence_groups[competence].append(example)
        
            positive_pairs = []
            rules_insufficient = {}
        
             #Forming positive pairs for each competence
            for competence, examples in competence_groups.items():
                #Check if there are enough examples to form pairs
                if len(examples) < min_examples:
                    missing_examples = min_examples - len(examples)
                    rules_insufficient[competence] = missing_examples
                    if remove_insufficient:
                        continue
                        
                 #Limit examples to a maximum of 40 per competence
                if len(examples) > max_examples_per_competence:
                    examples = examples[:max_examples_per_competence]
        
                num_examples = len(examples)
        
                 #Create pairs based on the number of examples
                if num_examples <= 10:
                    #Ensure each example is used 4 times as an anchor and 4 times as a positive
                    for i in range(num_examples):
                        positive_pairs.append((examples[i], examples[(i + 1) % num_examples]))
                        positive_pairs.append((examples[i], examples[(i + 2) % num_examples]))
                        positive_pairs.append((examples[i], examples[(i + 3) % num_examples]))
                        positive_pairs.append((examples[i], examples[(i + 4) % num_examples]))
        
                else:
                    #For competences with more than 10 examples, form a total of 40 pairs
                    random.shuffle(examples)
        
                    #Ensure we create exactly 40 pairs
                    pair_count = 0
                    while pair_count < 40:
                        anchor_idx = pair_count % num_examples
                        positive_idx = (anchor_idx + 1) % num_examples
                        positive_pairs.append((examples[anchor_idx], examples[positive_idx]))
                        pair_count += 1
        
        return positive_pairs, rules_insufficient
    
    def pair_data(self, positive_examples, negative_examples):
        """This function structures the data for the contrastive pre-training step. 
        It works by pairing all the initial positive_examples (i.e., anchors) with their
        positive pairs (i.e., other examples describing the same skill) and negative_pairs
        (i.e., sentences that do not contain any competences)"""

        #Empty lists for both positive and negative pairs
        self.tensorized_positives = []
        self.tensorized_negatives = []

        self.negative_examples = negative_examples

        #Define the size of the training data
        if self.training_size > 0:
            random.seed(self.seed)
            sample_size = min(int((self.training_size) * len(positive_examples)), len(positive_examples))
            positive_examples = random.sample(positive_examples, sample_size)
            sample_negative_size = min(int(self.training_size * len(negative_examples)), len(negative_examples))
            self.negative_examples = random.sample(negative_examples, sample_negative_size)

        #Form Pairs
        positive_pairs, _ = self.form_pairs(positive_examples)

        #Tensorize positive pairs
        for positive_pair in tqdm(positive_pairs, desc='Tensorizing positive pairs...'):
            tokens_anchor, rel_positions_anchor = positive_pair[0]['Tokens'], positive_pair[0]['relevant_mask']
            tokens_positive, rel_positions_positive = positive_pair[1]['Tokens'], positive_pair[1]['relevant_mask']
            #print(tokens_positive, rel_positions_positive)
            tensorized_anchor_df = self.tensorize_sentence(tokens_anchor, rel_positions_anchor)
            tensorized_positive_df = self.tensorize_sentence(tokens_positive, rel_positions_positive)
            #Append the list of dictionaries

            self.tensorized_positives.append({
                'anchor': {
                    'input_ids': tensorized_anchor_df[0],
                    'input_mask': tensorized_anchor_df[1],
                    'output_mask': tensorized_anchor_df[2],
                    'special_mask': tensorized_anchor_df[3]
                }, 
                'positive': {
                    'input_ids': tensorized_positive_df[0],
                    'input_mask': tensorized_positive_df[1],
                    'output_mask': tensorized_positive_df[2],
                    'special_mask': tensorized_positive_df[3]
                }
            })
            
        #Tensorize negative examples
        for negative_example in tqdm(self.negative_examples, desc='Tensorizing negative pairs...'):
            tokens_negative, rel_positions_negative = negative_example['Tokens'], negative_example['relevant_mask']
            tensorized_negative_def = self.tensorize_sentence(tokens_negative, rel_positions_negative)
            
            self.tensorized_negatives.append({
                'input_ids': tensorized_negative_def[0],
                'input_mask': tensorized_negative_def[1],
                'output_mask': tensorized_negative_def[2],
                'special_mask': tensorized_negative_def[3]
            })

        #Weak and strong negative_sampling
        if self.strong_negatives == True:
            return self.strong_negative_sampling(positive_pairs)
        else:
            return self.weak_negative_sampling()
        
    def weak_negative_sampling(self):
        """This function creates pairs for Contrastive Learning stage
        based on random allocation of the negatives"""

        number_negative_examples = self.number_negative_pairs
        negative_type = self.negative_type.lower()
        all_input_ids, all_input_masks, all_output_masks, all_special_masks = [], [], [], []
        all_pairs_ids, all_pairs_input_masks, all_pairs_output_masks, all_pairs_special_masks = [], [], [], []
        all_labels = []

        for i in tqdm(range(len(self.tensorized_positives)), desc='Creating Training Data with Weak Negative Sampling...'):
            labels = []

            #Retrieve tensorized anchor and its positive pair
            anchor = self.tensorized_positives[i]['anchor']
            positive = self.tensorized_positives[i]['positive']


            #Append anchor ids and masks
            all_input_ids.append(anchor['input_ids'])
            all_input_masks.append(anchor['input_mask'])
            all_output_masks.append(anchor['output_mask'])
            all_special_masks.append(anchor['special_mask'])

            #Positive pair
            pair_ids, pair_input_masks, pair_output_masks, pair_special_masks = [], [], [], []
            pair_ids.append(positive['input_ids'])
            pair_input_masks.append(positive['input_mask'])
            pair_output_masks.append(positive['output_mask'])
            pair_special_masks.append(positive['special_mask'])
            #The labels for the positive pair will be '1'
            labels.append(1)

            #Negative pair
            if negative_type == 'base':
                selected_negatives = random.choices(self.tensorized_negatives, k= number_negative_examples)

                for negative in selected_negatives:
                    pair_ids.append(negative['input_ids'])
                    pair_input_masks.append(negative['input_mask'])
                    pair_output_masks.append(negative['output_mask'])
                    pair_special_masks.append(negative['special_mask'])
                #The labels for the negative pairs will be '0'
                    labels.append(0)
                    
            elif negative_type == 'flip':
                
                selected_negatives = random.choices(self.tensorized_negatives, k= (number_negative_examples - 1))

                for negative in selected_negatives:
                    pair_ids.append(negative['input_ids'])
                    pair_input_masks.append(negative['input_mask'])
                    pair_output_masks.append(negative['output_mask'])
                    pair_special_masks.append(negative['special_mask'])
                #The labels for the negative pairs will be '0'
                    labels.append(0)
                    
                #Select one negative from list 
                flipped_output_mask_anchor = [1 if x == 0 else 0 if x == 1 else x for x in anchor['output_mask']]
                    
                pair_ids.append(anchor['input_ids'])
                pair_input_masks.append(anchor['input_mask'])
                pair_special_masks.append(anchor['special_mask'])
                pair_output_masks.append(flipped_output_mask_anchor)
                labels.append(0)
                
            #Append the all_ lists with the negatives and positives
            all_pairs_ids.append(pair_ids)
            all_pairs_input_masks.append(pair_input_masks)
            all_pairs_output_masks.append(pair_output_masks)
            all_pairs_special_masks.append(pair_special_masks)
            all_labels.append(labels)

        #convert all to the TensorDataset for easier access
        dataset = TensorDataset(
            torch.tensor(all_input_ids, dtype=torch.long),  
            torch.tensor(all_input_masks, dtype=torch.long),  
            torch.tensor(all_output_masks, dtype=torch.long), 
            torch.tensor(all_special_masks, dtype=torch.long),
            torch.tensor(all_pairs_ids, dtype=torch.long),  
            torch.tensor(all_pairs_input_masks, dtype=torch.long), 
            torch.tensor(all_pairs_output_masks, dtype=torch.long), 
            torch.tensor(all_pairs_special_masks, dtype=torch.long),
            torch.tensor(all_labels, dtype=torch.long) 
        )
        print(f'Overall {self.total_neg_padded} negative sequences were padded due to the exceeded tensor limit')
        return dataset
    
    def strong_negative_sampling(self, positive_examples):
        """This function works similarly as the above however,
        it utilises Meta's FAISS algorithm for the efficient 
        negatives selection (i.e., strong negatives)"""

        number_negative_examples = self.number_negative_pairs

        #probability for strong negatives to form 
        prob = self.strong_negative_prob
        
        #Initiate all the lists as before
        all_input_ids, all_input_masks, all_output_masks, all_special_masks = [], [], [], []
        all_pairs_ids, all_pairs_input_masks, all_pairs_output_masks, all_pairs_special_masks = [], [], [], []
        all_labels = []


        #Since finding positive pairs requires an encoder model to compare the embeddings we load the model vectorizer. 

        #Generate embeddings for anchor sentences 
        anchors = [entry[0] for entry in positive_examples]
        #assert whether number of anchors corresponds to the total number of pairs
        assert len(anchors) == len(positive_examples)
        anchor_embeddings = vectorizer_sentence(self.args, anchors, self, self.vectorizer)
        anchor_embeddings_norm = normalize_embeddings(anchor_embeddings)

        #Generate embeddings for negative_pairs
        negative_embeddings = vectorizer_sentence(self.args, self.negative_examples, self, self.vectorizer)
        
        nagative_embeddings_norm = normalize_embeddings(negative_embeddings)

        # No need to embed positive pair since it is already assigned 
        # Building a seracher object using FAISS
        # define the cluster size (as per documentation a square root of number of samples)
        res = faiss.StandardGpuResources()
        num_samples = len(nagative_embeddings_norm)
        nlist = int(round(math.sqrt(num_samples)))

        #Building a quantizer, since we need only to searches we use IndexIVFFlat to ensure balance between efficiency and quality
        quantizer = faiss.IndexFlatIP(768)
        index = faiss.IndexIVFFlat(quantizer, 768, nlist, faiss.METRIC_INNER_PRODUCT)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

        #train the index object (since it uses quantization)
        gpu_index.train(nagative_embeddings_norm)

        #add vectors to the index
        gpu_index.add(nagative_embeddings_norm)

        #iterate through all positives to find strongest negative matches
        for i in tqdm(range(len(positive_examples)), desc= 'Creating Training Data with Strong Negative Sampling...'):
            labels = []

            #Positive pair (the same as in weak sampling)
            anchor = self.tensorized_positives[i]['anchor']
            positive = self.tensorized_positives[i]['positive']
            all_input_ids.append(anchor['input_ids'])
            all_input_masks.append(anchor['input_mask'])
            all_output_masks.append(anchor['output_mask'])
            all_special_masks.append(anchor['special_mask'])
            pair_ids, pair_input_masks, pair_output_masks, pair_special_masks = [], [], [], []
            pair_ids.append(positive['input_ids'])
            pair_input_masks.append(positive['input_mask'])
            pair_output_masks.append(positive['output_mask'])
            pair_special_masks.append(positive['special_mask'])
            labels.append(1)

            #Negative pair (FAISS)
            if random.random() < prob:
            #Convert anchor instance to the float
                anchor_instance_eb = np.asarray([anchor_embeddings_norm[i]]).astype(np.float32)
                _, strong_indices = gpu_index.search(anchor_instance_eb, number_negative_examples)
                negative_indices = strong_indices[0]
            else:
                negative_indices = random.choices(list(range(len(self.negative_examples))), k= number_negative_examples)

            #Append the negative examples
            for idx in negative_indices:
                negative_pair = self.tensorized_negatives[idx]
                pair_ids.append(negative_pair['input_ids'])
                pair_input_masks.append(negative_pair['input_mask'])
                pair_output_masks.append(negative_pair['output_mask'])
                pair_special_masks.append(negative_pair['special_mask'])
                labels.append(0)

            #Append general lists
            all_pairs_ids.append(pair_ids)
            all_pairs_input_masks.append(pair_input_masks)
            all_pairs_output_masks.append(pair_output_masks)
            all_pairs_special_masks.append(pair_special_masks)
            all_labels.append(labels)


        #Convert to a tensor dataset
        dataset = TensorDataset(
            torch.tensor(all_input_ids, dtype=torch.long),
            torch.tensor(all_input_masks, dtype=torch.long),
            torch.tensor(all_output_masks, dtype=torch.long),
            torch.tensor(all_special_masks, dtype=torch.long),            
            torch.tensor(all_pairs_ids, dtype=torch.long),
            torch.tensor(all_pairs_input_masks, dtype=torch.long),
            torch.tensor(all_pairs_output_masks, dtype=torch.long),
            torch.tensor(all_pairs_special_masks, dtype=torch.long),
            torch.tensor(all_labels, dtype=torch.long)
        )
        print(f'Overall {self.total_neg_padded} negative sequences were padded due to the exceeded tensor limit')
        return dataset

class BioTaggingModel(torch.nn.Module):
    def __init__(self, args, model, num_labels):

        super().__init__()
        self.base_encoder = model
        self.args = args
        self.device = args.device
        self.classifier = nn.Linear(self.base_encoder.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(self.args.dropout_sup)
        self.num_labels = num_labels
        

    def forward(self, inputs_ids=None, inputs_mask=None, labels_mask=None):        
        #create the representations, get the last hidden state
        encodings = self.base_encoder(inputs_ids, attention_mask = inputs_mask)[0]
        #Apply dropout 
        if self.args.dropout_sup > 0:
            encodings = self.dropout(encodings)
        logits = self.classifier(encodings)
        
        #Since CrossEntropyLoss from PyTorch expects the inputs in batches, there is no need to multiply by batch size like in the Contrastive Loss calculation 
        #batch_size = inputs_ids.shape[0]
        #training mode
        if labels_mask is not None:
            #Define loss function
            loss_fct = nn.CrossEntropyLoss(ignore_index = -100)
            #Calculate loss, the logits needs to be reshaped from [batch_size, seq_length, num_labels] --> [batch_size * seq_length, num_labels]
            loss = loss_fct(logits.view(-1, self.num_labels), labels_mask.view(-1))

            return loss

        return logits 

def tensorize_data(args, data, tokenizer):
    """
    This function tensorizes the sequences in respect to the assigned labels.
    """
    #the same upper limit is applied to the negative examples
    limit = args.limit
    tagging_type = args.tagging_type
    model_type = args.model_type.lower()

    input_ids_list = []
    input_masks_list = []
    words_ids_list = []
    label_masks_list = []

    tokens_list = [item['tokens'] for item in data]
    labels_list = [item['tags_skill'] for item in data]
    #Transform labels into numerical values if BIO task
    if tagging_type.lower() == 'bio':
        tag_mapping = {'O': 0, 'B': 2, 'I': 1}
        labels_list = [[tag_mapping[tag] for tag in label_example] for label_example in labels_list]

    for tokens, labels in tqdm(zip(tokens_list, labels_list), desc='Tensorising data...', total= len(tokens_list)):
        
        tokenized_inputs = tokenizer(
            tokens,
            max_length = limit,
            padding = 'max_length',
            truncation= True, 
            is_split_into_words=True,
            return_tensors= 'pt',
        )
        
        input_ids = tokenized_inputs['input_ids'].squeeze(0).tolist()
        input_mask = tokenized_inputs['attention_mask'].squeeze(0).tolist()
        word_ids = [-100 if word_id is None else word_id for word_id in tokenized_inputs.word_ids()]
        
        label_mask = [-100 if word_id == -100 else labels[word_id] for word_id in word_ids]

        assert len(input_ids) == limit
        assert len(input_mask) == limit
        assert len(word_ids) == limit
        assert len(label_mask) == limit
        

        input_ids_list.append(input_ids)
        input_masks_list.append(input_mask)
        label_masks_list.append(label_mask)
        words_ids_list.append(word_ids)
    

    dataset = TensorDataset(
        torch.tensor(input_ids_list, dtype=torch.long),
        torch.tensor(input_masks_list, dtype=torch.long),
        torch.tensor(label_masks_list, dtype=torch.long),
        torch.tensor(words_ids_list, dtype=torch.long)
    )

    return dataset

def predict(args, test_data, model, tokenizer, batch_size=32):
    """
    This function is used to make predictions for the test data
    """
    if args.model_type.lower() == 'bert':
        cls_token = '[CLS]'
        sep_token = '[SEP]'
    else: 
        cls_token = '<s>'
        sep_token = '</s>'
        
    predictions, true_labels, all_tokens = [], [], []
    
    #process the test samples
    dataset = tensorize_data(args, test_data, tokenizer)
    #Set the model to evaluation mode
    
    #tag_mapping_flip = {0: 'O', 1: 'B', 2: 'I'}

    #adjust the batch_size for distributed learning 
    batch_size = batch_size * max(1, args.available_gpus)

    #Configure a sampler
    predict_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)

    #And a dataloader
    dataloader = DataLoader(dataset, sampler= predict_sampler, batch_size= batch_size)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Making Predictions...'):
            #check for dataloader
            if len(dataloader) <= 0:
                print('dataloader has no examples')
                break 
            #Transfer to the GPU if possible
            batch = tuple(example.to(args.device) for example in batch)
            inputs = {
                'inputs_ids': batch[0],
                'inputs_mask': batch[1],
            }
            labels_mask = batch[2]
            words_ids = batch[3]
            #Labels mask are retrieved separately to ensure the model returns the logits and not loss. 
            logits = model(**inputs)

            if args.device != 'cpu':
                logits = logits.detach().cpu().numpy()
                labels_mask = labels_mask.cpu().numpy()
            else:
                logits = logits.detach().numpy()
                labels_mask = labels_mask.numpy()
                
            #For each item in a sequence
            for i, logit_seq in enumerate(logits):
                # print(f'indiovidual preds {logit_seq}')
                token_predictions = []
                original_tokens = []
                true_labels_seq = []
                
                #Convert to tokens
                tokens = tokenizer.convert_ids_to_tokens(batch[0][i])
                # print(tokens)
                
                #To track the ID to group sub-tokens
                previous_word_id = None

                for j, token_logit in enumerate(logit_seq):
                    #Get the word id to check for sub-tokens 
                    word_id = words_ids[i][j]
                    #print(f'current word id is {word_id}')
                    # print(f'current token {tokens[j]}')
                    #print(f'current label is {labels_mask[i][j]}')
                    if labels_mask[i][j] == -100:
                        continue
                    
                    if word_id == -100 or word_id == previous_word_id:
                        continue

                    pred_label = np.argmax(token_logit)

                    #Similarly like in the above only consider the prediction for the first sub-token in a sequence
                    if tokens[j] != sep_token and tokens[j] != cls_token:
                        token_predictions.append(pred_label)
                        original_tokens.append(tokens[j])
                        true_labels_seq.append(labels_mask[i][j])

                    #Set the previous word id to track sub-tokens
                    previous_word_id = word_id
                    
                    #print(f'token predictions {token_predictions}')

                #Append the total lists, make sure the predicted labels are reverted back to the original BIO tags
                predictions.append(token_predictions)
                true_labels.append(true_labels_seq)
                all_tokens.append(original_tokens)
                #print(f'overall predictions {predictions}')

        return all_tokens, predictions, true_labels

# def check_pred(predictions):
#     for i in predictions:
#         #print(f'original list {i}')
#         for idx, _ in enumerate(i):
#             if idx in range(1, len(i) - 1):
#                 if i[idx] == 0 and i[idx + 1] == 2:
#                     i[idx + 1] = 1
#                 if idx in range(1, len(i) - 1) and i[idx] == 2 and i[idx + 1] == 1 and i[idx - 1] == 0:
#                     i[idx] = 1 
#                     i[idx + 1] = 2
#         #print(f'new list: {i}')
#     return predictions 

def evaluation(args, test_data, model, tokenizer, final=False):
    """
    This function takes the outputs from predict and evaluates them using the evaluate module
    """
    
    all_tokens, predictions, true_labels = predict(args, test_data, model, tokenizer, batch_size=16)
    # control_pred = args.control_pred
    tagging_type = args.tagging_type
    
    #Control predictions e.g., prevent unexpected patterns such as I preceded O or I followed by B
    # if control_pred:
    #     predictions = check_pred(predictions)

    #Revert back to the original BIO labels 
    if tagging_type.lower() == 'bio':
        tag_mapping_flip = {0: 'O', 2: 'B', 1: 'I'}
        predictions = [[tag_mapping_flip[tag] for tag in sequence] for sequence in predictions]
        true_labels = [[tag_mapping_flip[tag] for tag in sequence] for sequence in true_labels]
    # print(predictions[0])
    # print(true_labels[0])
    results = seqeval.compute(predictions=predictions, references=true_labels)
    
    prediction_list = []
    for tokens, preds, true in zip(all_tokens, predictions, true_labels):
        dictionary = {
            'Tokens': tokens,
            'Predicted': preds,
            'True Class': true
            }
        prediction_list.append(dictionary)
    #print(results)
    overall_precision = results['overall_precision'] 
    overall_recall = results['overall_recall']
    overall_f1 = results['overall_f1']
    overall_accuracy = results['overall_accuracy']
    
    if not final:
        # with open (os.path.join(args.model_directory, 'results_.json'), 'w') as f:
        #     json.dump(prediction_list, f)
        return overall_f1, overall_precision, overall_recall
    else: 
        print(f'The model achieves F1 score of {overall_f1}, with {overall_precision} precision and {overall_recall} recall. The overall accuracy is: {overall_accuracy}')
    
        prediction_list = []
        for tokens, preds, true in zip(all_tokens, predictions, true_labels):
            dictionary = {
                'Tokens': tokens,
                'Predicted': preds,
                'True Class': true
            }
            prediction_list.append(dictionary)

        with open (os.path.join(args.model_directory, '_results.json'), 'w') as f:
            json.dump(prediction_list, f)