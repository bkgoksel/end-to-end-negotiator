import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from models.simple_mute import SimpleProposerModule

NUM_ITEMS = 3
MAX_ITEM_CT = 4
DIALOGUE_CHOICES = ['Deal', 'No Deal', '<selection>']
FINAL_CHOICES = ['<no_agreement>']
TEMPLATE_STR = "I want %s books, %s hats and %s balls. You can have the rest."

class TemplateAgent(object):
    def __init__(self, domain, word_dict, args, name="Template"):
        self.name = name
        self.human = False
        self.domain = domain
        self.num_choices = (MAX_ITEM_CT + 1)**NUM_ITEMS
        self.eps = args.eps

        self.word_dict = word_dict
        self.t = 0

        self.args = args

        vocab_size = len(word_dict)
        print("vocab size is ", vocab_size)
        word_embedding_size = args.nembed_word
        ctx_size = NUM_ITEMS * 2
        reader_hidden_size = args.nreader
        proposer_hidden_size = args.nproposer
        num_dialog_choices = self.num_choices + len(DIALOGUE_CHOICES)
        num_final_choices = self.num_choices + len(FINAL_CHOICES)

        self.model = SimpleProposerModule(
                vocab_size,
                word_embedding_size,
                ctx_size,
                reader_hidden_size,
                proposer_hidden_size,
                num_dialog_choices,
                num_final_choices)

        self.all_rewards = []
        self.word_dict = word_dict
        self.last_hidden_state = Variable(torch.zeros(reader_hidden_size))
        self.opt = optim.SGD(
                self.model.parameters(),
                lr=self.args.rl_lr,
                momentum=self.args.momentum,
                nesterov=(self.args.nesterov and self.args.momentum > 0))

    def process_context(self, context):
        ctx = [int(x) for x in context]
        ctx_tensor = torch.Tensor(ctx)
        return Variable(ctx_tensor)

    def feed_context(self, ctx):
        print("called feed context")
        self.ctx = self.process_context(ctx)
        self.item_counts =  self.ctx[::2]
        self.last_hidden_state = torch.zeros_like(self.last_hidden_state)
        self.logprobs = []
        self.proposal_validity = []

    def read(self, conversation_input):
        enc_input = Variable(torch.LongTensor(self.word_dict.w2i(conversation_input)))
        _ , self.last_hidden_state = self.model.read(enc_input, self.ctx)

    def write(self):
        choice_logits = self.model.propose(self.last_hidden_state, self.ctx)

        if self.eps < np.random.rand():
            # sample from logits
            logprob, choice = self.sample_proposal(choice_logits)
            self.logprobs.append(logprob)
        else:
            logprob, choice = choice_logits.max(0)
            self.logprobs.append(logprob)

        proposal = self.build_proposal(choice, True)
        self.proposal_validity.append(-1*float(self.is_valid_proposal(proposal))-1)
        utterance = self.fill_dialog_template(proposal)

        return utterance

    def choose(self):
        choice_logits = self.model.choose(self.last_hidden_state, self.ctx)
        if self.eps < np.random.rand():
            # sample from logits
            logprob, choice = self.sample_proposal(choice_logits)
            self.logprobs.append(logprob)
        else:
            logprob, choice = choice_logits.max(0)
            self.logprobs.append(logprob)

        proposal = self.build_proposal(choice, False)
        self.proposal_validity.append(-1*float(self.is_valid_proposal(proposal))-1)
        utterance = self.fill_choice_template(proposal)

        return utterance


    def sample_proposal(self, logits):
        choice = logits.multinomial(1)

        logprob = logits[choice]

        # print(logprob)
        # print(choice)
        return logprob, choice

    def fill_dialog_template(self, proposal):
        if proposal in DIALOGUE_CHOICES:
            return [proposal]
        proposal = ['no' if num == 0 else str(num) for num in proposal]
        # print(proposal)
        res = TEMPLATE_STR % tuple(proposal)
        return res.split()

    def fill_choice_template(self, proposal):
        if proposal in FINAL_CHOICES:
            return [proposal]
        left_choice = ['item%d=%d' % (i,c) for i,c in enumerate(proposal)]
        right_choice = ['item%d=%d' % (i,n-c) for i, (n, c) in enumerate(zip(self.item_counts, proposal))]
        return ' '.join(left_choice + right_choice).split()

    def build_proposal(self, choice, dialogue):
        choice_val = choice.data[0]
        if choice_val > self.num_choices:
            if dialogue:
                return DIALOGUE_CHOICES[choice_val - (self.num_choices)]
            return FINAL_CHOICES[choice_val - (self.num_choices)]

        allocation = []

        for _ in range(NUM_ITEMS):
            allocation.append(choice_val % (MAX_ITEM_CT+1))
            choice_val = choice_val//(MAX_ITEM_CT+1)
        return allocation

    def is_valid_proposal(self, proposal):
        if proposal in DIALOGUE_CHOICES or proposal in FINAL_CHOICES:
            return True

        for (proposed, count) in zip(proposal, self.item_counts):
            if proposed.data[0] > count:
                return False

        return True


    def update(self, agree, reward):
        self.t += 1

        reward = reward if agree else 0

        self.all_rewards.append(reward)

        # standardize the reward
        r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))

        # compute accumulated discounted reward
        g = Variable(torch.zeros(1, 1).fill_(r))
        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        logrewards = list(zip(self.logprobs, rewards, self.proposal_validity))
        if logrewards:
            loss = -logrewards[0][0]*logrewards[0][1] + logrewards[0][2]
            # estimate the loss using one MonteCarlo rollout
            for lp, r, v in logrewards[1:]:
                loss = loss - lp * r + v


            self.opt.zero_grad()
            loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.rl_clip)
        self.opt.step()
