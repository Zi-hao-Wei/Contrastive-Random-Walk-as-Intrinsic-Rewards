import numpy as np
from model import Model, RNDModel
from memory import Memory, Transition
import torch
from torch import device
from torch import from_numpy
from torch.optim import Adam


class Agent:
    def __init__(self, env, n_actions, n_states, n_encoded_features):

        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 5e-3
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_encoded_features = n_encoded_features
        self.max_steps = 200
        self.max_episodes = 1000
        self.target_update_period = 500
        self.mem_size = 100000
        self.env = env
        self.recording_counter = 0
        self.batch_size = 64
        self.lr = 0.001
        self.gamma = 0.98
        self.device = device("cuda")

        self.q_target_model = Model(self.n_states, self.n_actions).to(self.device)
        self.q_eval_model = Model(self.n_states, self.n_actions).to(self.device)
        self.q_target_model.load_state_dict(self.q_eval_model.state_dict())


        self.memory = Memory(self.mem_size)

        self.loss_fn = torch.nn.MSELoss()
        self.q_optimizer = Adam(self.q_eval_model.parameters(), lr=self.lr)
        self.number_times_action_selected = np.zeros(n_actions)
    
    

    def choose_action(self, state):

        exp = np.random.rand()
        if self.epsilon > exp:
            return np.random.randint(self.n_actions)

        else:
            state = np.expand_dims(state, axis=0)
            state = from_numpy(state).float().to(self.device)
            return np.argmax(self.q_eval_model(state).detach().cpu().numpy())

    def update_train_model(self):
        self.q_target_model.load_state_dict(self.q_eval_model.state_dict())

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0  # as no loss
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)

        x = states
        q_eval = self.q_eval_model(x).gather(dim=1, index=actions.long())
        with torch.no_grad():
            q_next = self.q_target_model(next_states)

            q_eval_next = self.q_eval_model(next_states)
            max_action = torch.argmax(q_eval_next, dim=-1)

            batch_indices = torch.arange(end=self.batch_size, dtype=torch.int32)
            target_value = q_next[batch_indices.long(), max_action] * (1 - dones)

            q_target = rewards + self.gamma * target_value
        loss = self.loss_fn(q_eval, q_target.view(self.batch_size, 1))

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        dqn_loss = loss.detach().cpu().numpy()


        return dqn_loss

    def run(self):

        total_global_running_reward = []
        global_running_reward = 0
        for episode in range(1, 1 + self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            for step in range(1, 1 + self.max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _, = self.env.step(action)
                episode_reward += reward
                total_reward = reward
                self.store(state, total_reward, done, action, next_state)
                dqn_loss = self.train()
                if done:
                    break
                state = next_state

                if (episode * step) % self.target_update_period == 0:
                    self.update_train_model()

            self.epsilon = self.epsilon - self.decay_rate if self.epsilon > self.min_epsilon else self.min_epsilon

            if episode == 1:
                global_running_reward = episode_reward
            else:
                global_running_reward = 0.99 * global_running_reward + 0.01 * episode_reward

            total_global_running_reward.append(global_running_reward)
            print(f"EP:{episode}| "
                    f"DQN_loss:{dqn_loss:.3f}| "
                    f"EP_reward:{episode_reward}| "
                    f"EP_running_reward:{global_running_reward:.3f}| "
                    f"Epsilon:{self.epsilon:.2f}| "
                    f"Memory size:{len(self.memory)}")
            if episode % 50 == 0:
                self.save_weights()

        return total_global_running_reward

    def store(self, state, reward, done, action, next_state):
        state = from_numpy(state).float().to(self.device)
        reward = torch.Tensor([reward]).to(self.device)
        done = torch.Tensor([done]).to(self.device)
        action = torch.Tensor([action]).to(self.device)
        next_state = from_numpy(next_state).float().to(self.device)
        self.memory.add(state, reward, done, action, next_state)

    def unpack_batch(self, batch):

        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).to(self.device).view(self.batch_size, self.n_states)
        actions = torch.cat(batch.action).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device).view(self.batch_size, self.n_states)
        dones = torch.cat(batch.done).to(self.device)
        actions = actions.view((-1, 1))
        return states, actions, rewards, next_states, dones

    def save_weights(self):
        torch.save(self.q_eval_model.state_dict(), "weights.pth")

    def load_weights(self):
        self.q_eval_model.load_state_dict(torch.load("weights.pth", map_location="cpu"))

    def set_to_eval_mode(self):
        self.q_eval_model.eval()
