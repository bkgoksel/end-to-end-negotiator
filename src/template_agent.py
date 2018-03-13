class TemplateAgent(Agent):
    def __init__(self, domain, name="Template"):
        self.name = name
        self.human = False
        self.domain = domain
        self.model = SimpleProposerModule()
        self.all_rewards = []

    def feed_context(self, ctx):
        self.ctx = ctx
        self.ctx_h = self.model.process_context(ctx)
        self.logprobs = []

    def read(self, conversation_input):
        encoded_input = self.encode(conversation_input)
        self.last_conversation_embedding = self.model.read(encoded_input, self.ctx_h)

    def write(self):
        choice_logits = self.model.propose(self.last_conversation_embedding, self.my_last_proposal, self.ctx_h)

        if epsilon:
            # sample from logits
            proposal, logprob = choose(choice_logits)
            logprob = None
            self.logprobs.append(logprob)
        else:
            proposal = choose_max(choice_logits)

        # TODO:penalize invalid proposal
        self.my_last_proposal = proposal
        utterance = self.fill_template(proposal)

        return

    def choose(self):
        choice_logits = self.model.choose(self.last_conversation_embedding, self.my_last_proposal, self.ctx_h)
        if self.args.eps < np.random.rand():
            choose_max()


    def update(self, agree, reward):
        self.t += 1

        reward = reward if agree else 0
        # TODO: Penalize convo length?
        # TODO: Penalize bad proposals

        self.all_rewards.append(reward)

        # standardize the reward
        r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))

        # compute accumulated discounted reward
        g = Variable(torch.zeros(1, 1).fill_(r))
        rewards = g * (self.args.gamma**np.arange(len(self.logprobs)))

        loss = 0
        # estimate the loss using one MonteCarlo rollout
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.rl_clip)
        self.opt.step()
