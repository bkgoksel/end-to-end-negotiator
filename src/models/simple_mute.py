import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SimpleProposerModule(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, ctx_size, reader_hidden_size, proposer_hidden_size, num_dialog_choices, num_final_choices):
        super(SimpleProposerModule, self).__init__()
        # word embedding
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_size)

        #reader GRU
        self.reader = nn.GRU(input_size=word_embedding_size+ctx_size,
                            hidden_size=reader_hidden_size, batch_first=True)

        # self.last_hidden_state = None
        self.writer = nn.Sequential(
                nn.Linear(reader_hidden_size + ctx_size, proposer_hidden_size),
                nn.Tanh(),
                nn.Linear(proposer_hidden_size, num_dialog_choices),
                nn.Softmax(dim=0))
        # don't nead a writer GRU
        # don't need attention GRU
        #selection NN?
        self.selector = nn.Sequential(
                nn.Linear(reader_hidden_size + ctx_size, proposer_hidden_size),
                nn.Tanh(),
                nn.Linear(proposer_hidden_size, num_final_choices),
                nn.Softmax(dim=0))


    """
    Takes in an input utterance, outputs an embedding of the conversation state
    """
    def read(self, conversation_input, context_input):

        embedding = self.word_embedding(conversation_input)
        # print(embedding.shape)

        #CONCATENATE WORD INPUT AND CONTEXT
        context_input_expanded = context_input.expand((embedding.shape[0], context_input.shape[0]))

        embedding = torch.cat([embedding, context_input_expanded],1)
        # print(embedding.shape)

        out, last_hidden_state = self.reader(embedding.unsqueeze(0))

        # print(out.shape)
        # print(last_hidden_state.shape)
        return (out.squeeze(0), last_hidden_state.squeeze(0).squeeze(0))

    """
    Takes in the context embedding and latest conversation embedding, outputs a distribution over all possible proposals. Samples one.
    """
    def propose(self, conversation_input, context_input):
        # print(conversation_input.shape)
        # print(context_input.shape)
        # context_input_expanded = context_input.expand((conversation_input.shape[0], context_input.shape[0]))        
        proposer_input = torch.cat((conversation_input, context_input), 0)
        logits = self.writer(proposer_input) #
        # print(proposer_input.shape)
        # print(logits.shape)
        return logits

    """
    Takes in the context embedding and latest conversation embedding, outputs final choice.
    """
    def choose(self, conversation_input, context_input):
        selector_input = torch.cat([conversation_input, context_input], 0)
        logits = self.selector(selector_input) #
        return logits
