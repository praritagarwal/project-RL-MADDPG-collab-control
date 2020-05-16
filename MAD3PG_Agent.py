#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


from random import sample
from collections import deque
from Models import actor
from Models import critic
from Categorical_Distributions import projected_prob_batch2_torch


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


class Agent():
    def __init__(self, n_states = 24, n_actions = 2, actor_hidden = 256, 
                 critic_hidden = 600, seed = 0, roll_out = 5, replay_buffer_size = 1e6, 
                 replay_batch = 128, lr_actor = 1e-4,  lr_critic = 1e-4, epsilon = 0.3, 
                 epsilon_decay_rate = 0.999, tau = 1e-3,  gamma = 1, 
                 update_interval = 4, noise_fn = np.random.normal, 
                 vmin = -10, vmax = 10, n_atoms = 51, n_agents = 2):
        
        self.n_agents = 2
        self.n_states = n_states
        self.n_actions = n_actions
        self.actor_hidden = actor_hidden # hidden nodes in the 1st layer of actor network
        self.critic_hidden = critic_hidden # hidden nodes in the 1st layer of critic network
        self.seed = seed
        self.roll_out = roll_out # roll out steps for n-step bootstrap; taken to be same as in D4PG paper
        self.replay_buffer = replay_buffer_size
        self.replay_batch = replay_batch # batch of memories to sample during training
        self.lr_actor = lr_actor 
        self.lr_critic = lr_critic 
        self.epsilon = epsilon # to scale the noise before mixing with the actions; same as in D4PG paper
        self.epsilon_decay_rate = epsilon_decay_rate
        self.tau = tau # for soft updates of the target networks
        self.gamma = gamma 
        # note that we want the reacher to stay in goal position as long as possible
        # thus keeping gamma = 1 will ecourage the agent to increase its holding time
        self.update_every = update_interval # steps between successive updates
        self.noise = noise_fn # noise function; 
        # Note D4PG paper reported that 
        # using normal distribution instead of OU noise does not affect performance
        # will also experiment with OU noise if the need arises
        self.vmin = vmin
        self.vmax = vmax
        self.n_atoms = n_atoms
        self.delta = (vmax - vmin)/(n_atoms - 1)
        self.zi = torch.linspace(self.vmin, self.vmax, self.n_atoms).view(-1,1).to(device)
        # in numpy using linspace is much slower than the following way of doing it
        # but in torch its a little bit faster 
        # I guess that this is due to the time it takes to convert from numpy to torch tensors
        # self.zi = torch.from_numpy(np.array([ vmin + ii*self.delta for ii in range(self.n_atoms)])).view(-1,1).float().to(device)
        
        # In terminal states, we will massage the target prob. dist. such that it is one for the atom
        # having value zero and zero for all others
        # for this we need to know the index of the atom with zero value in self.zi
        self.zero_atom_pos = np.floor(-self.vmin/self.delta).astype(int)
        
        # discounts to be applied at each step of roll_out
        self.discounts = torch.tensor([self.gamma**powr 
                                       for powr in range(self.roll_out)]).double().view(-1,1).to(device)
        
        self.local_actors = [actor(self.n_states, self.n_actions, 
                                   self.actor_hidden, self.seed).to(device) \
                             for _ in range(self.n_agents)]
        
        # output of local critic network should be log_softmax
        self.local_critics = [critic(self.n_states, self.n_actions, 
                                     self.n_atoms, self.critic_hidden, 
                                     self.seed, output = 'logprob').to(device) \
                              for _ in range(self.n_agents)]
        
        self.target_actors = [actor(self.n_states, self.n_actions, 
                                    self.actor_hidden, self.seed).to(device) \
                              for _ in range(self.n_agents)]
        
        # target critic should output probabilities
        self.target_critics = [critic(self.n_states, self.n_actions, 
                                      self.n_atoms, self.critic_hidden, 
                                      self.seed, output = 'prob').to(device) \
                               for _ in range(self.n_agents) ]
        
        # initialize target_actor and target_critic weights to be 
        # the same as the corresponding local networks
        # Then instantiate the optimizers for the local actors and the local critics
        self.actor_optims = []
        self.critic_optims = []
        for idx in range(self.n_agents):
            for target_c_params, local_c_params in zip(self.target_critics[idx].parameters(),
                                                       self.local_critics[idx].parameters()):
                target_c_params.data.copy_(local_c_params.data)
            
            for target_a_params, local_a_params in zip(self.target_actors[idx].parameters(),
                                                       self.local_actors[idx].parameters()):
                target_a_params.data.copy_(local_a_params.data)
                
            # optimizers for the local actor and local critic
            self.actor_optims.append(torch.optim.Adam(self.local_actors[idx].parameters(),
                                                      lr = self.lr_actor))
            self.critic_optims.append(torch.optim.Adam(self.local_critics[idx].parameters(),
                                                       lr = self.lr_critic))
        
        # loss function
        self.criterion = nn.KLDivLoss(reduction = 'batchmean')
        
        # steps counter to keep track of steps passed between updates
        self.t_step = 0
        
        # replay memory 
        self.memory = ReplayBuffer(self.replay_buffer, self.n_states, 
                                   self.n_actions, self.roll_out, self.n_agents)
    
    def act(self, states):
        # convert states to a torch tensor and move to the device
        # for the multiagent case we will get a vstack of states
        # unsqueeze at index 1 to convert the state for each agent into a batch of size 1
        states = torch.from_numpy(states).unsqueeze(1).float().to(device)
        actions_list = []
        with torch.no_grad():
            for idx in range(self.n_agents):
                self.local_actors[idx].eval()
                actions = self.local_actors[idx](states[idx]).cpu().detach().numpy()
                noise = self.epsilon*self.noise(size = actions.shape)
                actions = np.clip(actions + noise, -1, 1)[0]
                actions_list.append(actions)
                self.local_actors[idx].train()
        actions_array = np.array(actions_list)        
        return actions_array
            
    def step(self, new_memories):
        # new memories is a batch of tuples
        # each tuple consists of (n-1)-steps of state, action, reward, done and the n-state
        # here n is the roll_out length
        self.memory.add(new_memories)
        
        # update the networks after every self.update_every steps
        # make sure to check that the replay_buffer has enough memories
        self.t_step = (self.t_step+1)%self.update_every
        if self.t_step == 0 and self.memory.__len__() > 2*self.replay_batch:
            self.learn()
            self.epsilon = max(self.epsilon_decay_rate*self.epsilon, 0.1 )
    
    def learn(self):
        # sample a batch of memories from the replay buffer
        states_0, actions_0, rewards, dones, states_fin = self.memory.sample(self.replay_batch)
        
        states_0 = torch.from_numpy(states_0).float().to(device)
        actions_0 = torch.from_numpy(actions_0).float().to(device)
        states_fin = torch.from_numpy(states_fin).float().to(device)
        rewards = torch.from_numpy(rewards).to(device)
        dones = torch.from_numpy(dones).to(device)
        
        # collect the target actions of all the agents in the final state
        # we need every agents action to pass to their respective target_critics
        t_actions_fin = []
        for idx in range(self.n_agents):
            # get an action for the n-th state from the target actor
            self.target_actors[idx].eval()
            with torch.no_grad():
                t_actions_fin.append(self.target_actors[idx](states_fin[:,idx]))
            self.target_actors[idx].train()
        t_actions_fin = torch.cat(t_actions_fin, dim = 1).to(device)
        
        
        # Compute the accumalated n_step_rewards 
        n_step_rewards = torch.matmul(rewards, self.discounts)
        
        # train the i-th agents critic
        # get the target probs for the n-th state 
        for idx in range(self.n_agents):
            self.target_critics[idx].eval()
            with torch.no_grad():
                # target critic directly outputs the probabilities 
                target_probs = self.target_critics[idx](states_fin.view(self.replay_batch,-1),
                                                        t_actions_fin)
            self.target_critics[idx].train()
            # note that in terminal states, we want the target distribution has to such that
            # the atom having 0 value has a prob 1 and all others are 0
            # Thus when done = 1, we will have to massage the target prob to be 1 at 0 value atom
            target_probs = target_probs*(1-dones) # the zeros the rows corresponding to dones
            target_probs[:,self.zero_atom_pos]+=dones.view(-1)
            projected_probs = projected_prob_batch2_torch(self.vmin, self.vmax, 
                                                          self.n_atoms, 
                                                          self.gamma**(self.roll_out), 
                                                          n_step_rewards[:,idx],
                                                          target_probs, self.replay_batch)
            
            # train the local critic
            self.critic_optims[idx].zero_grad()
            # get a Q_val dist. for the beginning state and action from the local critic
            local_log_probs = self.local_critics[idx](states_0.view(self.replay_batch,-1), 
                                                      actions_0)
            # compute the local critic's loss
            loss_c = self.criterion(local_log_probs, projected_probs)
            # can I just write loss_c = - torch.sum(projected_probs*local_log_probs)/self.replay_batch
            loss_c.backward()
            # clip grad norms as suggested in the D4PG paper
            # note that gradient clipping should be done after loss.backward() and 
            # before optimizer.step()
            # for example see Rahul's answer in the following stackoverflow post
            # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
            torch.nn.utils.clip_grad_norm_(self.local_critics[idx].parameters(), 1)
            self.critic_optims[idx].step()
            
            # now train the local actors
            self.actor_optims[idx].zero_grad() 
            
            # Now get the local_action of each agent for the initial state
            # remember to detach the actions of all the agents except the current one
            local_actions = []
            for idx2 in range(self.n_agents):
                if idx2 != idx:
                    with torch.no_grad():
                        local_a = self.local_actors[idx2](states_0[:,idx2]).detach()
                else:
                    local_a = self.local_actors[idx2](states_0[:,idx2])
                local_actions.append(local_a)
            local_actions = torch.cat(local_actions, dim = 1).to(device)
            assert local_actions.shape == (self.replay_batch, self.n_agents*self.n_actions),            'local actions does not have correct shape.'
            # local_a = self.local_actors[idx](states_0[:,idx])
            # get the Q_value for the initial state and local_a
            # this gives the actor's loss
            # apply torch.exp() to convert the critic's output into probabilities from log_prob
            probs = torch.exp(self.local_critics[idx](states_0.view(self.replay_batch, -1), 
                                                      local_actions))
            loss_a = -torch.matmul(probs, self.zi).mean()
            loss_a.backward()
            # clip grad norms as suggested in the D4PG paper
            # note that gradient clipping should be done after loss.backward() and 
            # before optimizer.step()
            # for example see Rahul's answer in the following stackoverflow post
            # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
            torch.nn.utils.clip_grad_norm_(self.local_actors[idx].parameters(), 1)
            self.actor_optims[idx].step()
        
        # apply soft updates to the target network
        self.update_target_networks()
      
    def update_target_networks(self):
        # update target actor
        for idx in range(self.n_agents):
            for params_target, params_local in zip(self.target_actors[idx].parameters(),
                                                   self.local_actors[idx].parameters()):
                updates = (1.0-self.tau)*params_target.data + self.tau*params_local.data 
                params_target.data.copy_(updates)
            
            # update target critic 
            for params_target, params_local in zip(self.target_critics[idx].parameters(), 
                                                   self.local_critics[idx].parameters()):
                updates = (1.0-self.tau)*params_target.data + self.tau*params_local.data 
                params_target.data.copy_(updates)
        


# In[ ]:


class ReplayBuffer():
    
    def __init__(self, buffer_size, n_states, n_actions, roll_out, n_agents):
        self.memory = deque(maxlen = int(buffer_size))
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.roll_out = roll_out # roll_out = 1 corresponds to a single step
        
        # length of an array containg a single memory of any one player
        self.experience_length = 2*n_states+n_actions+roll_out+1 
            
    def add(self, experience_tuple):
        self.memory.append(experience_tuple)
    
    def sample(self, batch_size):
        batch = np.array(sample(self.memory, batch_size))
        
        expected_batch_shape = (batch_size, self.n_agents, self.experience_length)
        
        assert batch.shape == expected_batch_shape,         'Shape of the batch is not same as expected.'        ' Got: {}, expected: {}!'.format(batch.shape, expected_batch_shape)
        
        states0_batch = batch[:,:,:self.n_states] # shape = (batch_size, n_agents, n_states)
        actions0_batch =        batch[:,:, self.n_states: self.n_states+self.n_actions].reshape(batch_size, -1)
        # shape = (batch_size, n_agents*n_actions)
        assert actions0_batch.shape == (batch_size, self.n_agents*self.n_actions),         'actions0 shape is incorrect'
        
        rewards_batch =        batch[:,:,self.n_states+self.n_actions:self.n_states+self.n_actions+self.roll_out] # shape = (batch_size, n_agents, roll_out)
        dones = batch[:,0,self.n_states+self.n_actions+self.roll_out:self.n_states+self.n_actions+self.roll_out+1]
        # shape = (batch_size, 1)
        states_fin_batch = batch[:,:,self.n_states+self.n_actions+self.roll_out+1:] # shape = (batch_size, n_agents, n_states)
        
        return  states0_batch, actions0_batch, rewards_batch, dones, states_fin_batch
        
        
       
    
    def __len__(self):
        return len(self.memory)

