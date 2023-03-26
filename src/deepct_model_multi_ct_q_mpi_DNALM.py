"""
DeepCT architecture without Sigmoid layer 
for multiple cell type per position computation at once
with quantitative features (TODO: Add our names).
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch import mean
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from dnalm.src.gena_lm.modeling_bert import *
import dnalm.src.gena_lm.modeling_bert as modeling_bert
from transformers import AutoConfig

# take care about "private" variables defined in the modeling_bert
_CHECKPOINT_FOR_DOC = modeling_bert._CHECKPOINT_FOR_DOC
_CONFIG_FOR_DOC = modeling_bert._CONFIG_FOR_DOC
_TOKENIZER_FOR_DOC = modeling_bert._TOKENIZER_FOR_DOC

@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, bert_config,
                n_cell_types,
                sequence_embedding_length,
                cell_type_embedding_length,
                final_embedding_length,
                n_genomic_features,
                dropout_rate=0.2,
            ):
        bert_config = AutoConfig.from_pretrained(bert_config)
        super().__init__(bert_config)
        # self.num_labels = config.num_labels
        self.config = bert_config

        self.bert = BertModel(bert_config)

        ######## from qDeepCT #######
        self._n_cell_types = n_cell_types
        self.n_genomic_features = n_genomic_features

        CLS_embedding_length = bert_config.hidden_size
        
        self.sequence_net = nn.Sequential(
            nn.Linear(CLS_embedding_length, sequence_embedding_length),
            nn.ReLU(inplace=True),
        )

        self.cell_type_net = nn.Sequential(
            nn.Linear(n_cell_types, cell_type_embedding_length),
        )

        self.seq_regressor = nn.Sequential(
            nn.Linear(sequence_embedding_length, sequence_embedding_length),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(sequence_embedding_length),
            nn.Linear(sequence_embedding_length, sequence_embedding_length),
            nn.ReLU(inplace=True),
            nn.Linear(sequence_embedding_length, sequence_embedding_length),
            nn.ReLU(inplace=True),
            nn.Linear(sequence_embedding_length, n_genomic_features),
            # sigmoid turned off for loss numerical stability
            # nn.Sigmoid(),
        )

        self.ct_regressor = nn.Sequential(
            nn.Linear(
                sequence_embedding_length + cell_type_embedding_length,
                final_embedding_length,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(final_embedding_length),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(final_embedding_length),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(final_embedding_length),
            nn.Linear(final_embedding_length, n_genomic_features),
            # sigmoid turned off for loss numerical stability
            # nn.Sigmoid(),
        )


        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        self.dropout = nn.Dropout(dropout_rate)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        batch_size = input_ids.size(0)

        cell_type_one_hots = torch.eye(self._n_cell_types).to(input_ids.device)

        # sequence_out = self.conv_net(sequence_batch)
        
        seq_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = seq_outputs[1]
        sequence_out = self.dropout(pooled_output)


        # reshaped_sequence_out = sequence_out.view(
        #     sequence_out.size(0), 960 * self._n_channels
        # )
        
        # print (f"sequence_out.shape {sequence_out.shape}")

        sequence_embedding = self.sequence_net(sequence_out)

        # Repeat each sequence embedding to fit cell type embeddings.
        # E.g., with 2 cell types, [seq0_emb, seq1_emb, seq2_emb] becomes
        # [seq0_emb, seq0_emb, seq1_emb, seq1_emb, seq2_emb, seq2_emb]
        repeated_sequence_embedding = sequence_embedding.repeat_interleave(
            repeats=self._n_cell_types, dim=0
        )

        # Repeat cell type embeddings to fit sequence embeddings.
        # E.g., with batch size of 3, 2 cell types, and [ct0_emb, ct1_emb] cell type
        # embeddings, the embeddings will be converted to
        # [ct0_emb, ct1_emb, ct0_emb, ct1_emb, ct0_emb, ct1_emb].
        cell_type_embeddings = self.cell_type_net(cell_type_one_hots).repeat(
            batch_size, 1
        )
        sequence_and_cell_type_embeddings = torch.cat(
            (repeated_sequence_embedding, cell_type_embeddings), 1
        )

        # view mean_positional_prediction to shape it as [batch_size, 1, n_genomic_features]
        mean_positional_prediction = self.seq_regressor(sequence_embedding).view(
            batch_size, 1, self.n_genomic_features
        )
        # view ct_deviations_prediction to shape it as [batch_size, _n_cell_types, n_genomic_features]
        ct_deviations_prediction = self.ct_regressor(
            sequence_and_cell_type_embeddings
        ).view(batch_size, self._n_cell_types, -1)
        predict = torch.cat((ct_deviations_prediction, mean_positional_prediction), 1)
        return predict

    def get_cell_type_embeddings(self):
        """Retrieve cell type embeddings learned by the model."""
        device = next(self.parameters()).device
        with torch.no_grad():
            all_cell_types = torch.eye(self._n_cell_types).to(device)
            embeddings = self.cell_type_net(all_cell_types)
        return embeddings.detach().cpu()

def get_optimizer(lr):
    """
    The optimizer and the parameters with which to initialize the optimizer.
    At a later time, we initialize the optimizer by also passing in the model
    parameters (`model.parameters()`). We cannot initialize the optimizer
    until the model has been initialized.
    """
    # Option 1:
    # return (torch.optim.SGD, {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})

    # Option 2:
    return (torch.optim.Adam, {"lr": lr, "weight_decay": 1e-6})
