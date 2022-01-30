import itertools
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from dataset import TSPFileReader

class AntQAlgorithm(object):
    def __init__(self, 
        number_of_agents, 
        learning_rate = 0.1, 
        discount_factor = 0.5, 
        action_choice_rule = 'pseudo-random-proportional',
        reinforcement_strategy = 'iteration-best', 
        utility_weight = 1.0,
        cost_weight = 2.0,
        exploration_level = 0.95, 
        reinforcement_factor = 10.0,
        max_iterations_to_improve = 1000, 
        max_iterations = 1000):
        self.number_of_agents = number_of_agents
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        if action_choice_rule not in ['pseudo-random', 'pseudo-random-proportional', 'random-propotional']:
            raise ValueError('Invalid action_choice_rule: {:s}'.format(action_choice_rule))
        self.action_choice_rule = action_choice_rule
        if reinforcement_strategy not in ['global-best', 'iteration-best']:
            raise ValueError('Invalid reinforcement_strategy: {:s}'.format(reinforcement_strategy))
        self.reinforcement_strategy = reinforcement_strategy
        self.utility_weight = utility_weight
        self.cost_weight = cost_weight
        self.exploration_level = exploration_level
        self.valid_states = {}
        self.initial_state = np.array([])
        self.current_state = np.array([])
        self.tours = {}
        self.distance_matrix = np.array([[]])
        self.action_choice_matrix = np.array([[]])
        self.q_values = np.array([[]])
        self.best_tours_lengths = np.array([])
        self.global_best_tour = []
        self.global_best_length = np.infty
        self.iteration_best_tour = []
        self.iteration_best_length = np.infty
        self.reinforcement_factor = reinforcement_factor
        self.max_iterations_to_improve = max_iterations_to_improve
        self.max_iterations = max_iterations
        self.iterations_without_improvement = 0
        self.step = 0
    
    def reset(self):
        self.reset_valid_states()
        self.reset_tours()

    def reset_tours(self):
        self.tours = dict(zip(range(self.number_of_agents), [[] for _ in range(self.number_of_agents)]))
        self.iteration_best_tour = []
        self.iteration_best_length = np.infty

    def reset_valid_states(self):
        self.valid_states = dict(zip(range(self.number_of_agents), [[] for _ in range(self.number_of_agents)]))
        states = set(range(n))
        starting_states = np.random.choice(list(states), self.number_of_agents, replace=True)
        for k in range(self.number_of_agents):
            self.current_state[k] =  starting_states[k]
            self.initial_state[k] = starting_states[k]
            self.valid_states[k] = states - set([starting_states[k]])
        return

    def initialize(self):
        # Initialization phase
        n = self.distance_matrix.shape[0]
        initial_q_value = (n - 1) / (np.sum(self.distance_matrix))
        self.q_values = initial_q_value * np.ones((n, n))
        self.update_action_choice_matrix()
        return

    def set_problem(self, distance_matrix):
        assert distance_matrix.shape[0] == distance_matrix.shape[1]
        self.distance_matrix = distance_matrix
        n = distance_matrix.shape[0]
        if n < self.number_of_agents:
            self.number_of_agents = n
        # Set valid states and current state per agent
        self.initial_state = -1 * np.ones((self.number_of_agents, ), dtype=int)
        self.current_state = -1 * np.ones((self.number_of_agents, ), dtype=int)
        self.best_tours_lengths = np.infty * np.ones((self.number_of_agents, ))
        self.global_best_tour = []
        self.global_best_length = np.infty
        return

    def transition_state(self, agent_id, r):
        valid_states = sorted(list(self.valid_states[agent_id]))
        q = np.random.rand()
        idx = np.argmax(self.action_choice_matrix[r, valid_states])
        max_val = self.action_choice_matrix[r, valid_states[idx]]
        if self.action_choice_rule != 'random-proportional' and \
            q < self.exploration_level:
            s = valid_states[idx]
        else:
            if self.action_choice_rule == 'pseudo-random':
                s = np.random.choice(list(valid_states))
            elif self.action_choice_rule == 'pseudo-random-proportional' or \
                self.action_choice_rule == 'random-proportional':
                probabilities = self.action_choice_matrix[r, valid_states] / \
                    np.sum(self.action_choice_matrix[r, valid_states])
                s = np.random.choice(list(valid_states), 1, replace=False, p=probabilities)[0] 
            else:
                pass
        return s, max_val

    def update_q_value(self, src_state, dst_state, use_reinforcement=True, arg_valid_states=None):
        if arg_valid_states is None:
            valid_states = list(set(range(self.q_values.shape[0])) - set([dst_state]))
        else:
            valid_states = list(arg_valid_states)
        if not use_reinforcement:
            correction_value = np.max(self.q_values[dst_state, valid_states])
            q_value = (1.0 - self.learning_rate) * self.q_values[src_state, dst_state] + \
                self.learning_rate * self.discount_factor * correction_value
        else:
            correction_value = np.max(self.q_values[dst_state, valid_states])
            reinforcement_value = self.reinforcement_matrix[src_state, dst_state]
            q_value = (1.0 - self.learning_rate) * self.q_values[src_state, dst_state] + \
                self.learning_rate * (self.discount_factor * correction_value + reinforcement_value)
            # print('({:02d}, {:02d}): corr_val = {}, q_val = {}'.format(src_state, dst_state, correction_value, q_value))
        self.q_values[src_state, dst_state] = q_value
        self.update_action_choice(src_state, dst_state)
        return q_value, correction_value

    def update_action_choice(self, src_state, dst_state):
        self.action_choice_matrix[src_state, dst_state] = \
            (self.q_values[src_state, dst_state] ** self.utility_weight) / \
            (self.distance_matrix[src_state, dst_state] ** self.cost_weight)

    def update_action_choice_matrix(self):
        local_matrix = self.distance_matrix.copy()
        local_matrix[np.eye(local_matrix.shape[0]).astype(bool)] = np.spacing(1)
        self.action_choice_matrix = (self.q_values ** self.utility_weight) / \
            (local_matrix ** self.cost_weight)

    def update_best_tour(self):
        lengths = np.zeros((self.number_of_agents, ))
        for k in range(self.number_of_agents):
            # Get length of the tour done by agent k
            for e in self.tours[k]:
                _, _, l_k = e
                lengths[k] += l_k
        # Update iteration best tour
        k_ib = np.argmin(lengths)
        self.iteration_best_tour = self.tours[k_ib]
        self.iteration_best_length = lengths[k_ib]
        # Check if there was any improvement
        if self.iteration_best_length < self.global_best_length:
            self.iterations_without_improvement = 0
            self.global_best_tour = self.iteration_best_tour
            self.global_best_length = self.iteration_best_length
        else:
            self.iterations_without_improvement += 1
        return

    def compute_reinforcement_matrix(self):
        # The delayed reinforcement matrix is a function of the lengths of all tours
        # Global best
        if self.reinforcement_strategy == 'global-best':
            self.reinforcement_matrix = np.zeros(self.distance_matrix.shape)
            for r, s, _ in self.global_best_tour:
                self.reinforcement_matrix[r, s] = self.reinforcement_factor / self.global_best_length
        elif self.reinforcement_strategy == 'iteration-best':
            self.reinforcement_matrix = np.zeros(self.distance_matrix.shape)
            for r, s, _ in self.iteration_best_tour:
                self.reinforcement_matrix[r, s] = self.reinforcement_factor / self.iteration_best_length
        else:
            pass
        return

    def pre_update(self):
        # This is the step in which agents build their tours. The tour of agent k is stored in
        # Tour k . Given that local reinforcement is always null, only the next state evaluation is used
        # to update AQ-values. */
        n = self.distance_matrix.shape[0]
        for i in range(n):
            #print()
            if i < n - 1:
                for k in range(self.number_of_agents):
                    # Choose the next city s_k according to formula (1)
                    r_k = self.current_state[k]
                    s_k, max_val = self.transition_state(k, r_k)
                    self.valid_states[k] = self.valid_states[k] - {s_k}
                    if i == n - 2:
                        self.valid_states[k].add(self.initial_state[k])
                    # Update q-value
                    q_value, learned_val = self.update_q_value(r_k, s_k, use_reinforcement=False, arg_valid_states=self.valid_states[k])
                    # Update tour
                    self.tours[k].append((r_k, s_k, self.distance_matrix[r_k, s_k]))
                    self.current_state[k] = s_k
                    # print('i = {:03d}, k = {:02d}: e = ({:02d}, {:02d}), m_val = {:}, l_val = {:}, q_val = {:}.'.format(
                    #     i, k, r_k, s_k, max_val, learned_val, q_value
                    # ))
            else:
                # In this cycle all agents go back to the initial city
                for k in range(self.number_of_agents):
                    r_k = self.current_state[k]
                    s_k = self.initial_state[k]
                    # Update tour
                    self.tours[k].append((r_k, s_k, self.distance_matrix[r_k, s_k]))
                    self.current_state[k] = s_k
        # Update action choice
        # self.update_action_choice_matrix()
        # Update best tour (iteration and global best tours)
        self.update_best_tour()
        return

    def update(self):
        # In this step delayed reinforcement is computed and AQ-values are updated using formula
        # (2), in which the next state evaluation term γ ·Max AQ(r k1 ,z) is null for all z */ 
        self.compute_reinforcement_matrix()
        n = self.distance_matrix.shape[0]
        # Update q-values with reinforcement
        # # Option 1: update just based on the best tour
        # best_tour = self.iteration_best_tour
        # if self.reinforcement_strategy == 'global-best':
        #     best_tour = self.global_best_tour
        # for r, s, _ in best_tour:
        #     self.update_q_value(r, s, use_reinforcement=True)
        # # Option 2: update all tours
        # for k, tour in self.tours.items():
        #     for r, s, _ in tour:
        #         self.update_q_value(r, s, use_reinforcement=True)
        # Option 3: update all edges
        for r, s in itertools.product(range(n), range(n)):
            self.update_q_value(r, s, use_reinforcement=True)
        self.update_action_choice_matrix()
        return

    def iterate(self):
        # New iteration
        self.step += 1
        # Reset lists
        self.reset()
        # Step 2: Build new tours
        self.pre_update()
        # Step 3: Update q-values
        self.update()
        # Check termination condition
        done = self.step == self.max_iterations or \
            self.iterations_without_improvement == self.max_iterations_to_improve
        print('Iteration {:03d}, best length = {:07.4f} '.format(self.step, self.global_best_length))
        sys.stdout.flush()
        return done

    def solve(self):
        # Step 1: initialize
        self.initialize()

        done = False
        while not done:
            done = self.iterate()
            # # New iteration
            # self.step += 1
            # # Reset lists
            # self.reset()
            # # Step 2: Build new tours
            # self.pre_update()
            # # Step 3: Update q-values
            # self.update()
            # # Check termination condition
            # done = self.step == self.max_iterations or \
            #     self.iterations_without_improvement == self.max_iterations_to_improve
            # print('Iteration {:03d}, best length = {:07.4f} '.format(self.step, self.global_best_length))
            # sys.stdout.flush()
        return self.global_best_tour, self.global_best_length


def add_arrow(line, size=15, color=None):
    if color is None:
        color = line.get_color()
        alpha = line.get_alpha()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    line.axes.annotate('',
        xytext=(xdata[0], ydata[0]),
        xy=(xdata[1], ydata[1]),
        arrowprops=dict(arrowstyle="->", color=color, alpha=alpha),
        size=size
    )

def del_arrows(ax):
    for i, an in reversed(list(enumerate(ax.texts))):
        if an._text == '':
            an.remove()

if __name__ == "__main__":
    # Generate a random matrix
    np.random.seed(1)

    graph = 'bays29.tsp'
    display_iterations = True

    tsp_reader = TSPFileReader(graph)
    distance_matrix = np.array(tsp_reader.dist_matrix)
    n = distance_matrix.shape[0]

    # Instantiate solver
    ant_q = AntQAlgorithm(n, max_iterations=500)
    # Set problem
    ant_q.set_problem(distance_matrix)

    x = [coordinates[0] for coordinates in tsp_reader.cities_tups]
    y = [coordinates[1] for coordinates in tsp_reader.cities_tups]
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle('Ant-Q - Traveling Salesman Problem: {:s}'.format(graph[:-4].upper()), fontsize=16)
    ax1 = axs[0]
    delta_x = max(x) - min(x)
    delta_y = max(y) - min(y)
    ax1.set_xlim(min(x) - 0.1*delta_x, max(x) + 0.1*delta_x)
    ax1.set_ylim(min(y) - 0.1*delta_y, max(y) + 0.1*delta_y)
    ax1.set_ylabel('y [m]')
    ax1.set_xlabel('x [m]')
    ax1.margins(0.05)
    # ax1.set_aspect('equal')
    ax1.set_title('Route', fontsize=14)
    ax1.scatter(x, y, marker='o', facecolors='none', edgecolors='blue', s=40)
    lines_set1 = []
    for i in range(n):
        ax1.text(x[i] + 0.01*delta_x, y[i] + 0.01*delta_y, '{:02d}'.format(i + 1))
        line, = ax1.plot([], [], color='blue', alpha=0.30)
        lines_set1.append(line)

    ax2 = axs[1]
    ax2.set_xlim(0, 5)
    if graph == 'bays29.tsp':
        ax2.set_ylim(2015, 2200)
    elif graph == 'oliver30.tsp':
        ax2.set_ylim(420, 500)
    else:
        pass
    ax2.margins(0.05)
    ax2.grid(True)
    # ax2.set_aspect('equal')
    ax2.set_title('Best route length', fontsize=14)
    ax2.set_ylabel('Length [m]')
    ax2.set_xlabel('Iteration')
    line2, = ax2.plot([], [], color='green')
    iterations = []
    lengths = []

    if display_iterations:
        done = False
        # Step 1: initialize
        ant_q.initialize()
        while not done:
            done = ant_q.iterate()
            global_best_route = ant_q.global_best_tour
            del_arrows(ax1)
            for k in range(n):
                i, j, _ = global_best_route[k]
                x_i, y_i = tsp_reader.cities_tups[i]
                x_j, y_j = tsp_reader.cities_tups[j]
                lines_set1[k].set_xdata([x_i, x_j])
                lines_set1[k].set_ydata([y_i, y_j])
                add_arrow(lines_set1[k])
            iterations.append(ant_q.step)
            lengths.append(ant_q.global_best_length)
            line2.set_xdata(iterations)
            line2.set_ydata(lengths)
            ax2.set_xlim(0, 5 + ant_q.step)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.savefig("figures/ant-q-{:04d}.png".format(ant_q.step))
    else:
        # Solve problem
        result = ant_q.solve()
        # Print result
        print('Length = {}'.format(result[1]))
        print('Result = \n{}'.format(result[0]))
