import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleProposerModule(modules.CudaModule):
    def __init__(self, device_id, args):
        super(ProposerModule, self).__init__(device_id)

        # word embedding
        self.word_embedding = nn.Embedding(len(self.word_dict), word_embedding_size)

        self.context_reader = nn.Linear()
        #reader GRU
        self.reader = nn.GRU(input_size=word_embedding_size+ctx_embedding_size,
                            hidden_size=reader_hidden_size)
        # self.last_hidden_state = None
        self.writer = Linear(PARAMS)
        # don't nead a writer GRU
        # don't need attention GRU

        #selection NN?


    def process_context(self, context):
        ctx = [int(x) for x in context.split()]
        ctx_tensor = torch.Tensor(ctx)
        return ctx_tensor
        

    """
    Takes in an input utterance, outputs an embedding of the conversation state
    """
    def read(self, conversation_input, context_input, vocab_size):
        embedding = self.word_embedding(conversation_input)

        #CONCATENATE WORD INPUT AND CONTEXT
        context_input_expanded = context_input.expand(embedding.size(0), context_input.size(0), context_input.size(1))
        embedding = torch.cat([embedding, context_input_expanded, 1]) 

        out, self.last_hidden_state = self.reader(embedding)

        return (out, self.last_hidden_state)
        pass

    """
    Takes in the context embedding and latest conversation embedding, outputs a distribution over all possible proposals. Samples one.
    """
    def propose(self, conversation_input, context_input):


        logits = self.writer(conversation_input, context_input) #  
        return logits
        #
        pass

    """
    Takes in the context embedding and latest conversation embedding, outputs final choice.
    """
    def choose(self, conversation_input, context_input):
        pass
