from data_manager import results
import json

# putting this here in case anyone runs plotter.py standalone
with open('src/lyapunov_adaptive_transformer/lyapunov_adaptive_transformer/config.json', 'r') as config_file:
    config = json.load(config_file)

results()

# to be called from LyAT node
def plot_results():

    with open('src/lyapunov_adaptive_transformer/lyapunov_adaptive_transformer/config.json', 'r') as config_file:
        config = json.load(config_file)

    results()
