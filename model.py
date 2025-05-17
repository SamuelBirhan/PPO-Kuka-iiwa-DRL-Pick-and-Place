import torch
import torch.nn as nn
import torch.nn.functional as F


def build_hidden_layer(input_dim, hidden_layers):
    """Build hidden layer."""
    hidden = nn.ModuleList([nn.Linear(input_dim, hidden_layers[0])])
    if len(hidden_layers) > 1:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        hidden.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    return hidden

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, shared_layers,
                 critic_hidden_layers=[], actor_hidden_layers=[],
                 seed=0, init_type=None):
        """Initialize parameters and build policy."""
        super(ActorCritic, self).__init__()
        self.init_type = init_type
        self.seed = torch.manual_seed(seed)
        self.sigma = nn.Parameter(torch.zeros(action_size))

        # Convolutional layers for processing images
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Calculate linear input size after convolutions
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[1])))
        linear_input_size = convh * convw * 32
        self.shared_layers = build_hidden_layer(input_dim=linear_input_size,
                                                hidden_layers=shared_layers)

        # Critic layers
        if critic_hidden_layers:
            self.critic_hidden = build_hidden_layer(input_dim=shared_layers[-1],
                                                    hidden_layers=critic_hidden_layers)
            self.critic = nn.Linear(critic_hidden_layers[-1], 1)
        else:
            self.critic_hidden = None
            self.critic = nn.Linear(shared_layers[-1], 1)

        # Actor layers
        if actor_hidden_layers:
            self.actor_hidden = build_hidden_layer(input_dim=shared_layers[-1],
                                                   hidden_layers=actor_hidden_layers)
            self.actor = nn.Linear(actor_hidden_layers[-1], action_size)
        else:
            self.actor_hidden = None
            self.actor = nn.Linear(shared_layers[-1], action_size)

        # Apply Tanh() to bound the actions
        self.tanh = nn.Tanh()

        # Initialize hidden layers
        if self.init_type is not None:
            self.shared_layers.apply(self._initialize)
            self.critic.apply(self._initialize)
            self.actor.apply(self._initialize)
            if self.critic_hidden is not None:
                self.critic_hidden.apply(self._initialize)
            if self.actor_hidden is not None:
                self.actor_hidden.apply(self._initialize)

    def _initialize(self, layer):
        """Initialize network weights."""
        if isinstance(layer, nn.Linear):
            if self.init_type == 'xavier-uniform':
                nn.init.xavier_uniform_(layer.weight.data)
            elif self.init_type == 'xavier-normal':
                nn.init.xavier_normal_(layer.weight.data)
            elif self.init_type == 'kaiming-uniform':
                nn.init.kaiming_uniform_(layer.weight.data)
            elif self.init_type == 'kaiming-normal':
                nn.init.kaiming_normal_(layer.weight.data)
            elif self.init_type == 'orthogonal':
                nn.init.orthogonal_(layer.weight.data)
            elif self.init_type == 'uniform':
                nn.init.uniform_(layer.weight.data)
            elif self.init_type == 'normal':
                nn.init.normal_(layer.weight.data)
            else:
                raise KeyError('Initialization type not recognized.')

    def forward(self, state):
        """Build a network that maps state -> (action, value)."""

        def apply_multi_layer(layers, x, f=F.leaky_relu):
            for layer in layers:
                x = f(layer(x))
            return x

        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        state = apply_multi_layer(self.shared_layers, state.view(state.size(0), -1))

        # Critic path
        v_hid = state
        if self.critic_hidden is not None:
            v_hid = apply_multi_layer(self.critic_hidden, v_hid)

        # Actor path
        a_hid = state
        if self.actor_hidden is not None:
            a_hid = apply_multi_layer(self.actor_hidden, a_hid)

        a = self.tanh(self.actor(a_hid))
        value = self.critic(v_hid).squeeze(-1)
        return a, value


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Model initialization
model = ActorCritic(state_size=(84, 84, 3), action_size=4, shared_layers=[256, 128])
model = model.to(device)

# Example input and forward pass
state = torch.randn(1, 3, 84, 84).to(device)  # Batch size of 1, 3 channels, 84x84 image
action, value = model(state)

print("Action:", action)
print("Value:", value)
