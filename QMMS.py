import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class CLTVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Central Limit Theorem Visualizer")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f5f6f8")
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TLabel", background="#f5f6f8", font=("Helvetica", 10))
        self.style.configure("Header.TLabel", font=("Helvetica", 14, "bold"), background="#f5f6f8", foreground="#2c3e50")
        self.style.configure("TButton", font=("Helvetica", 10, "bold"), background="#2980b9", foreground="white")
        self.style.map("TButton", background=[("active", "#3498db")])

        control_frame = ttk.Frame(self.root, padding=20)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        ttk.Label(control_frame, text="Simulation Controls", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 20))
        
        ttk.Label(control_frame, text="Population Distribution:").pack(anchor=tk.W, pady=(10, 2))
        self.dist_var = tk.StringVar(value="exponential")
        dist_cb = ttk.Combobox(control_frame, textvariable=self.dist_var, values=["exponential", "uniform"], state="readonly", font=("Helvetica", 10))
        dist_cb.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Sample Size (n):").pack(anchor=tk.W, pady=(10, 2))
        self.n_slider = tk.Scale(control_frame, from_=2, to=100, orient=tk.HORIZONTAL, bg="#f5f6f8", highlightthickness=0, fg="#2c3e50")
        self.n_slider.set(30)
        self.n_slider.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Number of Trials:").pack(anchor=tk.W, pady=(10, 2))
        self.trials_slider = tk.Scale(control_frame, from_=100, to=10000, resolution=100, orient=tk.HORIZONTAL, bg="#f5f6f8", highlightthickness=0, fg="#2c3e50")
        self.trials_slider.set(2000)
        self.trials_slider.pack(fill=tk.X, pady=(0, 20))
        
        run_btn = ttk.Button(control_frame, text="Generate Visualization", command=self.update_plots)
        run_btn.pack(fill=tk.X, ipady=5, pady=(10, 0))
        
        self.plot_frame = ttk.Frame(self.root, padding=10)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, self.axes = plt.subplots(1, 2, figsize=(9, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.update_plots()

    def update_plots(self):
        dist = self.dist_var.get()
        n = int(self.n_slider.get())
        trials = int(self.trials_slider.get())
        
        self.axes[0].clear()
        self.axes[1].clear()
        
        if dist == "uniform":
            population = np.random.uniform(0, 1, 100000)
        elif dist == "exponential":
            population = np.random.exponential(scale=1, size=100000)
            
        sample_means = [np.mean(np.random.choice(population, n)) for _ in range(trials)]
        
        self.axes[0].hist(population[:1000], bins=40, color="coral", edgecolor="white")
        self.axes[0].set_title(f"Population Profile ({dist.capitalize()})", fontsize=11, fontweight='bold', color="#34495e")
        self.axes[0].grid(axis='y', alpha=0.3)
        
        self.axes[1].hist(sample_means, bins=50, color="steelblue", edgecolor="white", density=True, label="Sample Distribution")
        self.axes[1].set_title(f"Sample Means Distribution (n={n})", fontsize=11, fontweight='bold', color="#34495e")
        self.axes[1].grid(axis='y', alpha=0.3)
        
        mu, sigma = np.mean(sample_means), np.std(sample_means)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        
        self.axes[1].plot(x, y, "r-", lw=2.5, label="Normal Curve")
        self.axes[1].legend(frameon=True, facecolor='white', edgecolor='none')
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = CLTVisualizerApp(root)
    root.mainloop()