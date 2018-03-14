import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleProposerModule(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, ctx_size, reader_hidden_size, proposer_hidden_size, num_dialog_choices, num_final_choices):
        # word embedding
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_size)

        #reader GRU
        self.reader = nn.GRU(input_size=word_embedding_size+ctx_size,
                            hidden_size=reader_hidden_size)

        # self.last_hidden_state = None
        self.writer = nn.Sequential(
                nn.Linear(reader_hidden_size + ctx_size, proposer_hidden_size),
                nn.Tanh(),
                nn.Linear(proposer_hidden_size, num_dialog_choices),
                nn.LogSoftmax(dim=0))
        # don't nead a writer GRU
        # don't need attention GRU
        #selection NN?
        self.selector = nn.Sequential(
                nn.Linear(reader_hidden_size + ctx_size, proposer_hidden_size),
                nn.Tanh(),
                nn.Linear(proposer_hidden_size, num_final_choices),
                nn.LogSoftmax(dim=0))


    """
    Takes in an input utterance, outputs an embedding of the conversation state
    """
    def read(self, conversation_input, context_input):
        embedding = self.word_embedding(conversation_input)

        #CONCATENATE WORD INPUT AND CONTEXT
        context_input_expanded = context_input.unsqueeze(0)
        embedding = torch.cat([embedding, context_input_expanded, 1])

        out, last_hidden_state = self.reader(embedding)

        return (out, last_hidden_state)

    """
    Takes in the context embedding and latest conversation embedding, outputs a distribution over all possible proposals. Samples one.
    """
    def propose(self, conversation_input, context_input):
        proposer_input = torch.cat([conversation_input, context_input], 0)
        logits = self.writer(proposer_input) #
        return logits

    """
    Takes in the context embedding and latest conversation embedding, outputs final choice.
    """
    def choose(self, conversation_input, context_input):
        selector_input = torch.cat([conversation_input, context_input], 0)
        logits = self.selector(selector_input) #
        return logits
