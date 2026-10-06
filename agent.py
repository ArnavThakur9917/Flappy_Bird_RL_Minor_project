# # Flappy Bird environment ko Python mein register/import karta hai.
# # Iske bina "FlappyBird-v0" environment available nahi hoga.
# import flappy_bird_gymnasium

# # Gymnasium RL environments ko create aur interact karne ke liye use hota hai.
# # Isse hum Flappy Bird environment ko create karenge.
# import gymnasium as gym

# # DQN neural network ko import kar rahe hain.
# # Ye state ko input lekar actions ke Q-values predict karega.
# from dqn import DQN

# # Experience Replay memory ko import kar rahe hain.
# # Ye agent ke past experiences ko store karegi.
# from experience_replay import ReplayMemory

# # itertools.count() ka use infinite episode loop ke liye karenge.
# # Training ko manually stop karne tak episodes chalte rahenge.
# import itertools

# # YAML file se hyperparameters read karne ke liye use hota hai.
# # Jaise learning rate, gamma, epsilon etc.
# import yaml

# import random

# # PyTorch main library hai neural network aur tensor operations ke liye.
# import torch

# # PyTorch ke neural-network related functions/classes ke liye use hota hai.
# import torch.nn as nn

# # Neural network ke weights update karne ke optimizers provide karta hai.
# import torch.optim as optim

# import os
# import argparse


# # Check karta hai ki Apple Metal GPU (MPS) available hai ya nahi.
# # Agar available hai to neural network MPS GPU par chalega.
# if torch.backends.mps.is_available():

#     # Apple GPU ko training ke liye select kar rahe hain.
#     device = "mps"

# # Agar CUDA GPU available hai to NVIDIA GPU use hoga.
# # CUDA generally NVIDIA GPUs ke liye use hota hai.
# elif torch.cuda.is_available():

#     # NVIDIA GPU ko training device select kar rahe hain.
#     device = "cuda"

# # Agar GPU available nahi hai to CPU use hoga.
# # Is case mein DQN normal processor par chalega.
# else:

#     # CPU ko training device select kar rahe hain.
#     device = "cpu"


# RUN_DIR = "runs"  
# os.makedirs(RUN_DIR, exist_ok=True)  


# # Agent class poore RL agent ko represent karegi.
# # Is class mein environment, DQN aur training logic handle hoga.
# class Agent:

#     # Constructor agent ko initialize karta hai.
#     # param_set mein training ke hyperparameters diye ja sakte hain.
#     def __init__(self, param_set):

#         # parameters.yaml file ko read mode mein open kar rahe hain.
#         # Is file mein DQN ke hyperparameters stored hain.
#         with open("parameters.yaml", "r") as file:

#             # YAML file ke data ko Python dictionary mein convert kar rahe hain.
#             # Ab hum parameters ko self.param_set ke through access kar sakte hain.
#             self.param_set = yaml.safe_load(file)
#             params = all_param_set[param_set] 


#             # Assign hyperparameters agnent ke pass inka acces rahe
#         self.alpha = params["alpha"]
#         self.gamma = params["gamma"]

#         self.epsilon_init = params["epsilon_init"]
#         self.epsilon_min = params["epsilon_min"]
#         self.epsilon_decay = params["epsilon_decay"]

#         self.replay_memory_size = params["replay_memory_size"]
#         self.mini_batch_size = params["mini_batch_size"]

#         self.reward_threshold = params["reward_threshold"]
#         self.network_sync_rate = params["network_sync_rate"]

#         self.loss_fn = nn.MSELoss()
#         self.optimizer = None # we use adam

#         #Ye training ki information/record save karne ke liye hai.
#         self.LOG_FILE = os.path.join(RUN_DIR,f"{self.para_set}.log")

#         #Ye trained neural network/model ke weights save karne ke liye hai
#         #Is .pt file mein trained PyTorch model ka data/weights save kiye ja sakte hain.
#         ##Baad mein model ko load karke bina starting se training kiye Flappy Bird play karwa sakte ho.
#         self.MODEL_FILE = os.path.join(RUN_DIR,f"{self.para_set}.pt")

#     # run() function actual Flappy Bird training/game loop chalata hai.
#     # is_training decide karta hai ki agent train karega ya sirf play karega.
#     def run(self, is_training=True, render=False):

#         # Flappy Bird environment create kar rahe hain.
#         # render=True hone par game window visually show hogi.
#         env = gym.make(
#             "FlappyBird-v0",
#             render_mode="human" if render else None
#         )

#         # Observation/state ke total features count karta hai.
#         # Ye number DQN ke input layer ka size hoga.
#         num_states = env.observation_space.shape[0]

#         # Environment mein possible actions ki count nikalta hai.
#         # Flappy Bird mein generally 2 actions hain: do nothing aur flap.
#         num_actions = env.action_space.n

#         # DQN neural network create kar rahe hain.
#         # num_states input aur num_actions output dimensions define karte hain.
#         policy_dqn = DQN(num_states, num_actions).to(device)

#         # Agar training mode on hai to Replay Memory create hogi.
#         # Ye agent ke experiences ko training ke liye store karegi.
#         if is_training:
#             epsilon = self.epsilon_init
#             # Maximum 10,000 experiences memory mein store hongi.
#             memory = ReplayMemory(self.replay_memory_size)


#             # Create the target network
#             # It has the same architecture as the policy network
#             target_dqn = DQN(num_states, num_actions).to(device)

#             # Copy all weights and biases from policy network to target network
#             # Initially both networks become identical
#             target_dqn.load_state_dict(policy_dqn.state_dict())

#             # traget network hme every episode main karna hoga
#             # iske liye step count krege 10 step main 1 epi hei
#             step = 0

#             self.optimizer = optim.Adam(policy_dqn.parameters(),lr=self.alpha)


#             # track best reward
#             best_reward = float("-inf")

#         else:
#             # if training nahi hogi to best policy load hogi = testing mode
#             policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))    
#             policy_dqn.eval()
#         # Infinite episodes start karte hain.
#         # itertools.count() 0, 1, 2, 3... continuously generate karta hai.
#         for episode in itertools.count():

#             # Environment ko new episode ke liye reset karta hai.
#             # Initial state aur additional information return hoti hai.
#             state, _ = env.reset()


#             #state convert into tensor
#             state = torch.tensor(state, dtype=torch.float,device=device)

#             # Current episode ka total reward initially zero hai.
#             # Har step ka reward is variable mein add hoga.
#             episode_reward = 0

#             # Starting mein bird alive hai. 
#             # terminated=True hone par episode end ho jayega.
#             terminated = False

#             # Jab tak bird game mein alive hai loop chalta rahega.
#             while (not terminated and episode_reward<self.self.reward_threshold):
#                 if is_training and random.random()<epsilon:
#                     # Abhi agent random action choose kar raha hai.
#                     # Baad mein yahan DQN se best action choose karwayenge.
#                     action = env.action_space.sample() # explore

#                     # Action convert into tensor
#                     action = torch.tensor(action, dtype=torch.float,device=device)
                    
#                 else:
#                     with torch.no_grad():
#                         action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploit
#                 # Selected action environment ko dete hain.
#                 # Environment next state, reward aur termination status return karta hai.
#                 next_state, reward, terminated, _, _ = env.step(action.item())

#                 # Current step ka reward episode ke total reward mein add karte hain.
#                 # Isse pata chalega ki agent ne complete episode mein kitna score kiya.
#                 episode_reward += reward

#                 # next_state convert into tensor
#                 next_state = torch.tensor(next_state, dtype=torch.float,device=device)

#                 # reward convert into tensor
#                 reward = torch.tensor(reward, dtype=torch.float,device=device)
            

#                 # Sirf training mode mein experience memory mein store karenge.
#                 # Experience mein state, action, next state, reward aur done information hoti hai.
#                 if is_training:

#                     # Current transition ko Replay Memory mein store kar rahe hain.
#                     # Ye baad mein random mini-batch lekar DQN training mein use hogi.
#                     memory.append(
#                         (state, action, reward, next_state, terminated)
#                     )
#                     steps +=1

#                 # Current state ko next state bana rahe hain.
#                 # Agle step mein agent isi updated state se decision lega.
#                 state = next_state

                

#             # Current episode ka number aur total reward print karta hai.
#             # Isse training ke progress ko terminal mein monitor kar sakte hain.
#             print(
#                 f"Episode = {episode + 1}, "
#                 f"Total Reward = {episode_reward}"
#             )



#             if is_training:

#                 # epsilon decay
#                 epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

#                 if episode_reward > best_reward:
#                     log_msg = f"best reward = {episode_reward} for episode = {episode+1}"

#                     with open(self.LOG_FILE,"a") as f:
#                         f.write(log_msg +"\n")
#                     torch.save(policy_dqn.state_dict(),self.MODEL_FILE)
#                     best_reward = episode_reward


#             # shink the network
#             if is_training and len(memory)>self.min_batch_size:
#                 # get sample
#                 mini_batch = memory.sample(self.mini_batch_size)

#                 # ab hum value optimize karege
#                 # Train/update the policy network using the sampled experiences
#                 self.optimize(mini_batch, policy_dqn, target_dqn)

#                 # Sync the target network after a fixed number of steps
#                 if steps > self.network_sync_rate:

#                     # Copy the latest weights and biases
#                     # from policy network to target network
#                     target_dqn.load_state_dict(policy_dqn.state_dict())

#                     # Reset the step counter
#                     steps = 0



#     def optimize(self, mini_batch, policy_dqn, target_dqn):

#     # ---------------------------------------------------------
#     # STEP 1: Get batch of experiences
#     # ---------------------------------------------------------
#     # mini_batch mein bahut saare experiences hain.
#     #
#     # Har experience:
#     # (state, action, next_state, reward, termination)
#     #
#     # zip(*mini_batch) sabhi experiences ko alag-alag groups
#     # mein divide karta hai.

#         states, actions, next_states, rewards, terminations = zip(*mini_batch)


#     # ---------------------------------------------------------
#     # Convert data into PyTorch tensors
#     # ---------------------------------------------------------

#     # Saare current states ko ek tensor mein convert karo
#         states = torch.stack(states)

#     # Saare actions ko ek tensor mein convert karo
#         actions = torch.stack(actions)

#     # Saare next states ko ek tensor mein convert karo
#         next_states = torch.stack(next_states)

#     # Saare rewards ko ek tensor mein convert karo
#         rewards = torch.stack(rewards)

#     # Termination ko tensor mein convert karo
#     # float() isliye kiya gaya hai taaki 0/1 ke form mein use ho sake.
#         terminations = torch.tensor(terminations).float().to(device)


#     # ---------------------------------------------------------
#     # STEP 2: Calculate TARGET Q-values
#     # ---------------------------------------------------------

#     # Target Q-value calculate karte time
#     # Target Network ke weights ko update nahi karna hai.
#     #
#     # Isliye torch.no_grad() use karte hain.

#         with torch.no_grad():

#         # Target Q-value:
#         #
#         # reward
#         # +
#         # (agar game khatam nahi hua)
#         # discount_factor × future Q-value
#         #
#         # target_dqn(next_states) se next state ke Q-values milenge.
#         # max(dim=1)[0] har next state ka highest Q-value leta hai.

#             target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]


#     # ---------------------------------------------------------
#     # STEP 3: Calculate CURRENT / PREDICTED Q-value
#     # ---------------------------------------------------------

#     # Current state ko Policy Network mein bhejo.
#     #
#     # Policy DQN har action ke liye Q-value deta hai.
#     #
#     # Example:
#     #
#     # FLAP     = 3.2
#     # NOTHING  = 2.1
#     #
#     # Lekin hume sirf wahi Q-value chahiye
#     # jo action actually liya gaya tha.

#         current_q = policy_dqn(states).gather(
#             dim=1,
#             index=actions.unsqueeze(dim=1)
#         ).squeeze()


#     # ---------------------------------------------------------
#     # STEP 4: Calculate LOSS
#     # ---------------------------------------------------------

#     # Current Q-value aur Target Q-value ko compare karo.
#     #
#     # Agar:
#     # current_q = 3.2
#     # target_q  = 4.6
#     #
#     # Difference = error
#     #
#     # Loss function isi error ko calculate karega.

#         loss = self.loss_fn(current_q, target_q)


#     # ---------------------------------------------------------
#     # STEP 5: Clear old gradients
#     # ---------------------------------------------------------

#     # Previous training step ke gradients ko remove karo.

#         self.optimizer.zero_grad()


#     # ---------------------------------------------------------
#     # STEP 6: Backpropagation
#     # ---------------------------------------------------------

#     # Loss ke basis par Policy Network ke
#     # gradients calculate karo.

#         loss.backward()


#     # ---------------------------------------------------------
#     # STEP 7: Update Policy Network
#     # ---------------------------------------------------------

#     # Adam optimizer gradients ka use karke
#     # Policy DQN ke weights update karega.

#         self.optimizer.step() 



# if __name__ == "__main__":

#     # Create ArgumentParser object
#     # Iska use command-line arguments lene ke liye hota hai.
#     # Example:
#     # python agent.py --train
#     parser = argparse.ArgumentParser(
#         description="Train or test model."
#     )

#     # --train naam ka argument add kar rahe hain.
#     # Agar user --train likhega, to args.train = True ho jayega.
#     parser.add_argument(
#         "--train",
#         help="Training mode",
#         action="store_true"
#     )

#     # Command-line arguments ko read/parse karo.
#     args = parser.parse_args()


#     # Agent object create karo.
#     # args.hyperparameters mein model ke hyperparameters honge.
#     # Example: learning rate, gamma, epsilon, etc.
#     dql = Agent(param_set=args.hyperparameters)


#     # Check karo ki user ne training mode select kiya hai ya nahi.
#     if args.train:

#         # Training mode
#         # is_training=True ka matlab model learn/update karega.
#         dql.run(is_training=True)

#     else:

#         # Testing / playing mode
#         # is_training=False ka matlab model learn nahi karega.
#         #
#         # render=True ka matlab Flappy Bird ka game
#         # screen par display hoga.
#         dql.run(
#             is_training=False,
#             render=True
#         )                     
#         # env.close() yahan intentionally nahi rakha hai.
#         # Training ko continuously run karne ke liye environment open rahega.






# =========================================================
# IMPORTS
# =========================================================

# ============================================================
# agent.py
# Flappy Bird - Deep Q-Network (DQN)
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import flappy_bird_gymnasium
import gymnasium as gym

from dqn import DQN
from experience_replay import ReplayMemory

import itertools
import yaml
import random
import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse


# ------------------------------------------------------------
# 2. SELECT DEVICE
# ------------------------------------------------------------
# If NVIDIA GPU is available -> CUDA
# If Apple GPU is available -> MPS
# Otherwise -> CPU
# ------------------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"

elif torch.backends.mps.is_available():
    device = "mps"

else:
    device = "cpu"


print(f"Using device: {device}")


# ------------------------------------------------------------
# 3. CREATE RUNS DIRECTORY
# ------------------------------------------------------------
# This folder will contain:
#
# .log -> training log
# .pt  -> trained PyTorch model
# ------------------------------------------------------------

RUN_DIR = "runs"

os.makedirs(RUN_DIR, exist_ok=True)


# ============================================================
# AGENT CLASS
# ============================================================

class Agent:

    # --------------------------------------------------------
    # 4. CONSTRUCTOR
    # --------------------------------------------------------

    def __init__(self, param_set):

        # Save parameter set name
        self.param_set = param_set

        # ----------------------------------------------------
        # Load parameters from parameters.yaml
        # ----------------------------------------------------

        with open("parameters.yaml", "r") as file:

            all_param_set = yaml.safe_load(file)

        # Select required parameter set
        params = all_param_set[param_set]

        # ----------------------------------------------------
        # DQN PARAMETERS
        # ----------------------------------------------------

        self.alpha = params["alpha"]

        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]

        self.epsilon_min = params["epsilon_min"]

        self.epsilon_decay = params["epsilon_decay"]

        # ----------------------------------------------------
        # Replay Memory parameters
        # ----------------------------------------------------

        self.replay_memory_size = params["replay_memory_size"]

        self.mini_batch_size = params["mini_batch_size"]

        # ----------------------------------------------------
        # Reward parameters
        # ----------------------------------------------------

        self.reward_threshold = params["reward_threshold"]

        # ----------------------------------------------------
        # Target network synchronization
        # ----------------------------------------------------

        self.network_sync_rate = params["network_sync_rate"]

        # ----------------------------------------------------
        # Loss function
        # ----------------------------------------------------

        self.loss_fn = nn.MSELoss()

        # Optimizer will be created during training
        self.optimizer = None

        # ----------------------------------------------------
        # Log file
        # ----------------------------------------------------

        self.LOG_FILE = os.path.join(
            RUN_DIR,
            f"{self.param_set}.log"
        )

        # ----------------------------------------------------
        # Model file
        # ----------------------------------------------------

        self.MODEL_FILE = os.path.join(
            RUN_DIR,
            f"{self.param_set}.pt"
        )


    # ========================================================
    # RUN METHOD
    # ========================================================

    def run(self, is_training=False, render=False):

        # ----------------------------------------------------
        # 5. CREATE FLAPPY BIRD ENVIRONMENT
        # ----------------------------------------------------

        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None
        )

        # ----------------------------------------------------
        # Get number of states
        # ----------------------------------------------------

        num_states = env.observation_space.shape[0]

        # ----------------------------------------------------
        # Get number of actions
        # ----------------------------------------------------

        num_actions = env.action_space.n

        print(f"Number of states  : {num_states}")
        print(f"Number of actions : {num_actions}")


        # ====================================================
        # 6. CREATE POLICY DQN
        # ====================================================
        #
        # Policy network:
        #
        # state
        #   ↓
        # policy_dqn
        #   ↓
        # Q-values
        #
        # It is the network that learns.
        # ====================================================

        policy_dqn = DQN(
            num_states,
            num_actions
        ).to(device)


        # ====================================================
        # TRAINING SETUP
        # ====================================================

        if is_training:

            print("\n==============================")
            print("       TRAINING MODE")
            print("==============================\n")


            # ------------------------------------------------
            # 7. EPSILON
            # ------------------------------------------------
            #
            # High epsilon:
            # More exploration
            #
            # Low epsilon:
            # More exploitation
            # ------------------------------------------------

            epsilon = self.epsilon_init


            # ------------------------------------------------
            # 8. REPLAY MEMORY
            # ------------------------------------------------

            memory = ReplayMemory(
                self.replay_memory_size
            )


            # ------------------------------------------------
            # 9. TARGET DQN
            # ------------------------------------------------
            #
            # Target network provides stable Q-values.
            # ------------------------------------------------

            target_dqn = DQN(
                num_states,
                num_actions
            ).to(device)


            # Initially target network = policy network
            target_dqn.load_state_dict(
                policy_dqn.state_dict()
            )


            # Target network is only used for prediction
            target_dqn.eval()


            # ------------------------------------------------
            # 10. NETWORK SYNC COUNTER
            # ------------------------------------------------

            steps = 0


            # ------------------------------------------------
            # 11. ADAM OPTIMIZER
            # ------------------------------------------------

            self.optimizer = optim.Adam(
                policy_dqn.parameters(),
                lr=self.alpha
            )


            # ------------------------------------------------
            # 12. BEST REWARD
            # ------------------------------------------------

            best_reward = float("-inf")

            best_episode = 0


            # ------------------------------------------------
            # TRAINING RUNS FOREVER
            # until you stop it manually
            # ------------------------------------------------

            episode_range = itertools.count()


        # ====================================================
        # TESTING SETUP
        # ====================================================

        else:

            print("\n==============================")
            print("        TESTING MODE")
            print("==============================\n")


            # ------------------------------------------------
            # Load trained model
            # ------------------------------------------------

            if not os.path.exists(self.MODEL_FILE):

                print("ERROR:")
                print(
                    f"Model file not found: {self.MODEL_FILE}"
                )

                print(
                    "\nFirst train the model using:"
                )

                print(
                    "py -3.12 agent.py "
                    "--param flappybirdv0 --train"
                )

                env.close()

                return


            print(
                f"Loading model: {self.MODEL_FILE}"
            )


            policy_dqn.load_state_dict(
                torch.load(
                    self.MODEL_FILE,
                    map_location=device
                )
            )


            # Put network in evaluation mode
            policy_dqn.eval()


            # ------------------------------------------------
            # IMPORTANT:
            # Testing = ONLY ONE EPISODE
            # ------------------------------------------------

            episode_range = range(1)


            # These are not needed for testing
            epsilon = 0

            memory = None

            target_dqn = None

            steps = 0

            best_reward = float("-inf")

            best_episode = 0


        # ====================================================
        # 13. EPISODE LOOP
        # ====================================================

        for episode in episode_range:


            # ------------------------------------------------
            # Reset environment
            # ------------------------------------------------

            state, _ = env.reset()


            # ------------------------------------------------
            # Convert state to PyTorch tensor
            # ------------------------------------------------

            state = torch.tensor(
                state,
                dtype=torch.float,
                device=device
            )


            # ------------------------------------------------
            # Episode reward starts from 0
            # ------------------------------------------------

            episode_reward = 0


            # ------------------------------------------------
            # Episode is initially not terminated
            # ------------------------------------------------

            terminated = False


            # =================================================
            # 14. PLAY ONE EPISODE
            # =================================================

            while (
                not terminated
                and episode_reward < self.reward_threshold
            ):


                # ------------------------------------------------
                # ACTION SELECTION
                # ------------------------------------------------
                #
                # Training:
                #
                # random.random() < epsilon
                #       ↓
                # random action
                #
                # Otherwise:
                #       ↓
                # DQN chooses best action
                #
                # Testing:
                #       ↓
                # always choose best action
                # ------------------------------------------------

                if (
                    is_training
                    and random.random() < epsilon
                ):

                    # --------------------------------------------
                    # EXPLORATION
                    # --------------------------------------------

                    action = env.action_space.sample()

                    action = torch.tensor(
                        action,
                        dtype=torch.long,
                        device=device
                    )


                else:

                    # --------------------------------------------
                    # EXPLOITATION
                    # --------------------------------------------

                    with torch.no_grad():

                        action = policy_dqn(
                            state.unsqueeze(dim=0)
                        ).squeeze().argmax()


                # ------------------------------------------------
                # Take action in Flappy Bird
                # ------------------------------------------------

                next_state, reward, terminated, _, _ = env.step(
                    action.item()
                )


                # ------------------------------------------------
                # Convert reward to tensor
                # ------------------------------------------------

                reward = torch.tensor(
                    reward,
                    dtype=torch.float,
                    device=device
                )


                # ------------------------------------------------
                # Convert next state to tensor
                # ------------------------------------------------

                next_state = torch.tensor(
                    next_state,
                    dtype=torch.float,
                    device=device
                )


                # ------------------------------------------------
                # Add reward to episode reward
                # ------------------------------------------------

                episode_reward += reward.item()


                # =================================================
                # 15. STORE EXPERIENCE
                # =================================================
                #
                # Experience:
                #
                # (state,
                #  action,
                #  next_state,
                #  reward,
                #  terminated)
                #
                # This is stored only during training.
                # =================================================

                if is_training:

                    memory.append(
                        (
                            state,
                            action,
                            next_state,
                            reward,
                            terminated
                        )
                    )

                    # Count environment steps
                    steps += 1


                # ------------------------------------------------
                # Move to next state
                # ------------------------------------------------

                state = next_state


            # =================================================
            # 16. EPISODE RESULT
            # =================================================

            print(
                f"Episode = {episode + 1}, "
                f"Total Reward = {episode_reward:.2f}"
            )


            # =================================================
            # 17. TRAINING AFTER EPISODE
            # =================================================

            if is_training:


                # ------------------------------------------------
                # Reduce epsilon
                # ------------------------------------------------
                #
                # epsilon becomes smaller over time.
                #
                # More exploration
                #       ↓
                # Less exploration
                #       ↓
                # More exploitation
                # ------------------------------------------------

                epsilon = max(
                    epsilon * self.epsilon_decay,
                    self.epsilon_min
                )


                # ------------------------------------------------
                # 18. CHECK BEST REWARD
                # ------------------------------------------------

                if episode_reward > best_reward:

                    # Update best reward
                    best_reward = episode_reward

                    # Save episode number
                    best_episode = episode + 1


                    # ------------------------------------------------
                    # Save information in log file
                    # ------------------------------------------------

                    log_msg = (
                        f"best reward = {episode_reward} "
                        f"for episode = {episode + 1}"
                    )


                    with open(
                        self.LOG_FILE,
                        "a"
                    ) as f:

                        f.write(
                            log_msg + "\n"
                        )


                    # ------------------------------------------------
                    # Save policy network
                    # ------------------------------------------------

                    torch.save(
                        policy_dqn.state_dict(),
                        self.MODEL_FILE
                    )


                    print(
                        f"*** New Maximum Reward = "
                        f"{best_reward:.2f} "
                        f"at Episode = "
                        f"{best_episode} ***"
                    )


                # =================================================
                # 19. REPLAY MEMORY TRAINING
                # =================================================
                #
                # We only train when replay memory contains
                # enough experiences.
                # =================================================

                if len(memory) >= self.mini_batch_size:


                    # ------------------------------------------------
                    # Randomly select mini-batch
                    # ------------------------------------------------

                    mini_batch = memory.sample(
                        self.mini_batch_size
                    )


                    # ------------------------------------------------
                    # Perform DQN optimization
                    # ------------------------------------------------

                    self.optimize(
                        mini_batch,
                        policy_dqn,
                        target_dqn
                    )


                # =================================================
                # 20. SYNCHRONIZE TARGET NETWORK
                # =================================================
                #
                # Target network is updated after a fixed number
                # of steps.
                # =================================================

                if steps >= self.network_sync_rate:


                    target_dqn.load_state_dict(
                        policy_dqn.state_dict()
                    )


                    # Reset counter
                    steps = 0


        # ====================================================
        # 21. TESTING RESULT
        # ====================================================

        if not is_training:

            print("\n==============================")
            print("       TESTING FINISHED")
            print("==============================")

            print(
                f"Testing Reward = "
                f"{episode_reward:.2f}"
            )


        # ----------------------------------------------------
        # Close environment
        # ----------------------------------------------------

        env.close()


    # ========================================================
    # OPTIMIZE METHOD
    # ========================================================

    def optimize(
        self,
        mini_batch,
        policy_dqn,
        target_dqn
    ):

        # ====================================================
        # 22. UNPACK MINI-BATCH
        # ====================================================
        #
        # Every experience contains:
        #
        # state
        # action
        # next_state
        # reward
        # termination
        # ====================================================

        states, actions, next_states, rewards, terminations = zip(
            *mini_batch
        )


        # ----------------------------------------------------
        # Convert states to tensor
        # ----------------------------------------------------

        states = torch.stack(states)


        # ----------------------------------------------------
        # Convert actions to tensor
        # ----------------------------------------------------

        actions = torch.stack(actions)


        # ----------------------------------------------------
        # Convert next states to tensor
        # ----------------------------------------------------

        next_states = torch.stack(next_states)


        # ----------------------------------------------------
        # Convert rewards to tensor
        # ----------------------------------------------------

        rewards = torch.stack(rewards)


        # ----------------------------------------------------
        # Convert termination flags to tensor
        # ----------------------------------------------------

        terminations = torch.tensor(
            terminations,
            dtype=torch.float,
            device=device
        )


        # ====================================================
        # 23. CALCULATE TARGET Q-VALUE
        # ====================================================
        #
        # Bellman equation:
        #
        # Target Q =
        #
        # Reward +
        # (1 - Done) * Gamma * Max Future Q
        #
        # If episode is terminated:
        #
        # Target Q = Reward
        #
        # because there is no future state.
        # ====================================================

        with torch.no_grad():

            target_q = (
                rewards
                + (1 - terminations)
                * self.gamma
                * target_dqn(next_states)
                .max(dim=1)[0]
            )


        # ====================================================
        # 24. CURRENT Q-VALUE
        # ====================================================
        #
        # Policy DQN predicts Q-value for every action.
        #
        # Example:
        #
        # Action 0 -> Q = 1.5
        # Action 1 -> Q = 3.2
        #
        # If actual action was 0:
        #
        # Current Q = 1.5
        #
        # gather() selects Q-value of action actually taken.
        # ====================================================

        current_q_values = policy_dqn(states)


        current_q = current_q_values.gather(
            dim=1,
            index=actions.unsqueeze(dim=1)
        ).squeeze()


        # ====================================================
        # 25. CALCULATE LOSS
        # ====================================================
        #
        # Loss tells us:
        #
        # How different is Current Q
        # from Target Q?
        # ====================================================

        loss = self.loss_fn(
            current_q,
            target_q
        )


        # ====================================================
        # 26. BACKPROPAGATION
        # ====================================================

        # Remove old gradients
        self.optimizer.zero_grad()


        # Calculate new gradients
        loss.backward()


        # Update policy DQN weights
        self.optimizer.step()


# ============================================================
# 27. MAIN PROGRAM
# ============================================================
#
# Training:
#
# py -3.12 agent.py --param flappybirdv0 --train
#
#
# Testing:
#
# py -3.12 agent.py --param flappybirdv0
#
# Testing will run ONLY ONE EPISODE.
# ============================================================

if __name__ == "__main__":


    # --------------------------------------------------------
    # Create command-line argument parser
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Train or test Flappy Bird DQN model."
    )


    # --------------------------------------------------------
    # Parameter set argument
    # --------------------------------------------------------

    parser.add_argument(
        "--param",
        default="flappybirdv0",
        help="Parameter set name from parameters.yaml"
    )


    # --------------------------------------------------------
    # Training argument
    # --------------------------------------------------------

    parser.add_argument(
        "--train",
        action="store_true",
        help="Training mode"
    )


    # --------------------------------------------------------
    # Read command-line arguments
    # --------------------------------------------------------

    args = parser.parse_args()


    # --------------------------------------------------------
    # Create Agent
    # --------------------------------------------------------

    dql = Agent(
        param_set=args.param
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    if args.train:

        dql.run(
            is_training=True,
            render=False
        )


    # --------------------------------------------------------
    # TESTING
    # --------------------------------------------------------

    else:

        dql.run(
            is_training=False,
            render=True
        )