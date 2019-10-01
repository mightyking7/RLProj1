import sys
from matplotlib import pyplot as plt

filename = sys.argv[1]

with open(filename, "r") as f:
    lines = f.readlines()

    sample_average = {
        "average_rs": [float(n) for n in lines[0].strip().split()],
        "average_best_action_taken": [float(n) for n in lines[1].strip().split()],
    }
    constant = {
        "average_rs": [float(n) for n in lines[2].strip().split()],
        "average_best_action_taken": [float(n) for n in lines[3].strip().split()],
    }
    #
    # assert (
    #     len(sample_average["average_rs"])
    #     == len(sample_average["average_best_action_taken"])
    #     == len(constant["average_rs"])
    #     == len(constant["average_best_action_taken"])
    #     == 10000
    # )

    fig, axes = plt.subplots(2, 1)

    axes[1].set_ylim([0.0, 1.0])

    axes[0].plot(sample_average["average_rs"], label="Sample Average")
    axes[1].plot(sample_average["average_best_action_taken"], label="Sample Average")

    axes[0].plot(constant["average_rs"], label="Constant Step Size")
    axes[1].plot(constant["average_best_action_taken"], label="Constant Step Size")

    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Average Reward over Time of Bandit Learners")
    axes[0].legend()

    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("% Best Action Taken")
    axes[1].set_title("How Often Bandit Learners Took Best Action")
    axes[1].legend()

    fig.show()

    input("Press Enter to continue...")