from models import modules
from models import dialog_model

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProposerModule(modules.CudaModule):
    def __init__(self, word_dict, item_dict, output_length, device_id, args):
        super(ProposerModule, self).__init__(device_id)
        self.word_dict = word_dict
        self.item_dict = item_dict
        # a bidirectional selection RNN
        # it will go through input words and generate by the reader hidden states
        # to produce a hidden representation
        self.sel_rnn = nn.GRU(
            input_size=args.nhid_lang + args.nembed_word,
            hidden_size=args.nhid_attn,
            bias=True,
            bidirectional=True)

        # mask for disabling special tokens when generating sentences
        self.special_token_mask = torch.FloatTensor(len(self.word_dict))

        # attention to combine selection hidden states
        self.attn = nn.Sequential(
            torch.nn.Linear(2 * args.nhid_attn, args.nhid_attn),
            nn.Tanh(),
            torch.nn.Linear(args.nhid_attn, 1)
        )

        # selection encoder, takes attention output and context hidden and combines them
        self.sel_encoder = nn.Sequential(
            torch.nn.Linear(2 * args.nhid_attn + args.nhid_ctx, args.nhid_sel),
            nn.Tanh()
        )
        # selection decoders, one per each item
        self.sel_decoders = nn.ModuleList()
        for i in range(output_length):
            self.sel_decoders.append(nn.Linear(args.nhid_sel, len(self.item_dict)))

    def forward(self, h, attn_h, ctx_h):
        # runs selection rnn over the hidden state h
        h, _ = self.sel_rnn(h, attn_h)

        # perform attention
        logit = self.attn(h).squeeze(1)
        prob = F.softmax(logit, dim=0).unsqueeze(1).expand_as(h)
        attn = torch.sum(torch.mul(h, prob), 0, keepdim=True)

        # concatenate attention and context hidden and pass it to the selection encoder
        h = torch.cat([attn, ctx_h], 2)
        h = self.sel_encoder.forward(h)

        # generate logits for each item separately
        logits = [decoder.forward(h).squeeze(0) for decoder in self.sel_decoders]
        return logits

class MuteModel(dialog_model.DialogModel):
    def __init__(self, word_dict, item_dict, context_dict, output_length, args, device_id):
        super(MuteModel, self).__init__(word_dict, item_dict, context_dict, output_length, args, device_id)
        self.writer = ProposerModule(word_dict, item_dict, output_length, device_id, args)

    def write(self, inpt, lang_h, ctx_h):
        # run a birnn over the concatenation of the input embeddings and
        # language model hidden states
        inpt_emb = self.word_encoder(inpt)
        h = torch.cat([lang_h, inpt_emb], 2)
        h = self.dropout(h)

        # runs selection rnn over the hidden state h
        attn_h = self.zero_hid(h.size(1), self.args.nhid_attn, copies=2)
        logits = self.writer(h, attn_h, ctx_h)
        return logits

    def score_sent(self, sent):
        raise NotImplementedError
        #index into specific parts of sentence and multiply
