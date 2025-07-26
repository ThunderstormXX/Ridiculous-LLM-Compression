# pruninghealing/logger.py
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

class Logger:
    def __init__(self, workspace_dir="./workspace"):
        self.workspace_dir = workspace_dir
        self.log_file = os.path.join(workspace_dir, "experiment_log.json")
        self.logs = []
        os.makedirs(workspace_dir, exist_ok=True)
        
    def log_step(self, step_data):
        """Log a single step/iteration"""
        step_data["timestamp"] = datetime.now().isoformat()
        self.logs.append(step_data)
        self._save_logs()
        
    def log_experiment(self, experiment_name, config, results):
        """Log complete experiment"""
        experiment_data = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": results
        }
        self.logs.append(experiment_data)
        self._save_logs()
        
    def plot_perplexity(self, save_path=None):
        """Plot perplexity over iterations"""
        if not self.logs:
            return
            
        iterations = []
        pre_ppl = []
        post_ppl = []
        
        for log in self.logs:
            if "step" in log:
                iterations.append(log["step"])
                pre_ppl.append(log.get("pre_train_perplexity", 0))
                post_ppl.append(log.get("post_train_perplexity", 0))
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, pre_ppl, 'r-', label='Pre-training')
        plt.plot(iterations, post_ppl, 'b-', label='Post-training')
        plt.xlabel('Iteration')
        plt.ylabel('Perplexity')
        plt.title('Perplexity During Pruning and Healing')
        plt.legend()
        plt.grid(True)
        
        if save_path is None:
            save_path = os.path.join(self.workspace_dir, "perplexity_plot.png")
        plt.savefig(save_path)
        plt.close()
        
    def get_summary(self):
        """Get experiment summary"""
        if not self.logs:
            return {}
            
        summary = {
            "total_steps": len([log for log in self.logs if "step" in log]),
            "final_perplexity": None,
            "best_perplexity": float('inf'),
            "total_layers_removed": 0
        }
        
        for log in self.logs:
            if "post_train_perplexity" in log:
                ppl = log["post_train_perplexity"]
                if ppl < summary["best_perplexity"]:
                    summary["best_perplexity"] = ppl
                summary["final_perplexity"] = ppl
            if "removed_layer" in log:
                summary["total_layers_removed"] += 1
                
        return summary
        
    def _save_logs(self):
        """Save logs to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
            
    def load_logs(self):
        """Load existing logs"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)
        return self.logs