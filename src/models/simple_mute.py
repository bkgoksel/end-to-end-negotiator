import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleProposerModule(modules.CudaModule):
    def __init__(self, device_id, args):
        super(ProposerModule, self).__init__(device_id)

    def process_context(self, context):
        pass

    """
    Takes in an input utterance, outputs an embedding of the conversation state
    """
    def read(self, conversation_input, context_input):
        pass

    """
    Takes in the context embedding and latest conversation embedding, outputs a distribution over all possible proposals. Samples one.
    """
    def propose(self, conversation_input, context_input):
        pass

    """
    Takes in the context embedding and latest conversation embedding, outputs final choice.
    """
    def choose(self, conversation_input, context_input):
        pass
