import tkinter as tk
from tkinter import ttk
from tkinter import colorchooser
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

mpl.rcParams['toolbar'] = 'toolbar2'
mpl.rcParams['backend'] = 'TkAgg'

class CLTVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Central Limit Theorem Visualizer")
        self.root.geometry("1180x740")
        
        self.COLOR_BG = "#1e222b" 
        self.COLOR_PANEL = "#282c34"
        self.COLOR_TEXT = "#abb2bf"
        self.COLOR_HEADER = "#ffffff"
        self.COLOR_ACCENT = "#4b6cb7"
        self.COLOR_HOVER = "#5c7ed8"
        
        self.root.configure(bg=self.COLOR_BG)

        root.option_add("*TCombobox*Listbox.background", self.COLOR_PANEL)
        root.option_add("*TCombobox*Listbox.foreground", "#ffffff")
        root.option_add("*TCombobox*Listbox.selectBackground", self.COLOR_ACCENT)
        root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.style.configure("TFrame", background=self.COLOR_BG)
        self.style.configure("Panel.TFrame", background=self.COLOR_PANEL)
        
        self.style.configure("TLabel", background=self.COLOR_PANEL, foreground=self.COLOR_TEXT, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), background=self.COLOR_PANEL, foreground=self.COLOR_HEADER)
        self.style.configure("Sub.TLabel", font=("Segoe UI", 10, "bold"), background=self.COLOR_PANEL, foreground=self.COLOR_HEADER)
        
        self.style.configure("TButton", 
                             font=("Segoe UI", 10, "bold"), 
                             background=self.COLOR_ACCENT, 
                             foreground="white", 
                             borderwidth=0,
                             focusthickness=0,
                             relief="flat",
                             padding=(10, 6))
        self.style.map("TButton", background=[("active", self.COLOR_HOVER)])

        self.style.configure("Toggle.TButton", 
                             font=("Segoe UI", 10, "bold"), 
                             background="#3a404c",        
                             foreground="#ffffff",        
                             borderwidth=0,
                             relief="flat",
                             padding=(2, 4))
        self.style.map("Toggle.TButton", 
                       background=[("active", self.COLOR_ACCENT)], 
                       foreground=[("active", "#ffffff")])

        self.style.configure("TCombobox", 
                             fieldbackground="#ffffff",
                             background=self.COLOR_PANEL,
                             foreground="#1e222b",
                             borderwidth=1, 
                             arrowcolor=self.COLOR_TEXT)
        self.style.map("TCombobox", 
                       fieldbackground=[("readonly", "#ffffff")],
                       foreground=[("readonly", "#1e222b")])

        self.sidebar_visible = True
        self.sidebar_width = 300
        self.is_animating = False

        self.side_container = ttk.Frame(self.root, style="TFrame")
        self.side_container.pack(side=tk.LEFT, fill=tk.Y)

        self.control_frame = ttk.Frame(self.side_container, padding=25, style="Panel.TFrame", width=self.sidebar_width)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.control_frame.pack_propagate(False)

        self.toggle_btn = ttk.Button(self.root, text="◀", width=3, style="Toggle.TButton")
        self.toggle_btn.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(self.control_frame, text="Simulation Controls", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 15))
        
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(self.control_frame, text="Population Distribution:", style="Sub.TLabel").pack(anchor=tk.W, pady=(5, 5))
        self.dist_var = tk.StringVar(value="exponential")
        dist_cb = ttk.Combobox(self.control_frame, textvariable=self.dist_var, values=["exponential", "uniform"], state="readonly")
        dist_cb.pack(fill=tk.X, pady=(0, 15))

        self.n_var = tk.IntVar(value=30)
        self.trials_var = tk.IntVar(value=2000)
        
        self.n_var.trace_add("write", lambda *args: self.validate_entry(self.n_var, 2, 100))
        self.trials_var.trace_add("write", lambda *args: self.validate_entry(self.trials_var, 100, 10000))
        
        self.n_slider = self.create_linked_control(self.control_frame, "Sample Size (n):", self.n_var)
        self.n_slider.configure(from_=2, to=100)

        self.trials_slider = self.create_linked_control(self.control_frame, "Number of Trials:", self.trials_var)
        self.trials_slider.configure(from_=100, to=10000, resolution=100)

        ttk.Label(self.control_frame, text="Graph Customization", style="Header.TLabel").pack(anchor=tk.W, pady=(20, 15))
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 15))
        
        self.pop_color_var = tk.StringVar(value="#ff7f50")
        self.sample_color_var = tk.StringVar(value="#4682b4")

        pop_color_frame = ttk.Frame(self.control_frame, style="Panel.TFrame")
        pop_color_frame.pack(fill=tk.X, pady=(5, 10))
        ttk.Label(pop_color_frame, text="Population Color:").pack(side=tk.LEFT)
        
        self.pop_preview = tk.Frame(pop_color_frame, width=22, height=22, bg=self.pop_color_var.get(), highlightthickness=1, highlightbackground="#ffffff")
        self.pop_preview.pack(side=tk.RIGHT, padx=(8, 0))
        self.pop_preview.pack_propagate(False)
        
        pop_color_btn = ttk.Button(pop_color_frame, text="Pick Color", style="TButton", width=11, command=lambda: self.pick_color(self.pop_color_var, self.pop_preview))
        pop_color_btn.pack(side=tk.RIGHT)

        sample_color_frame = ttk.Frame(self.control_frame, style="Panel.TFrame")
        sample_color_frame.pack(fill=tk.X, pady=(5, 25))
        ttk.Label(sample_color_frame, text="Sample Color:").pack(side=tk.LEFT)
        
        self.sample_preview = tk.Frame(sample_color_frame, width=22, height=22, bg=self.sample_color_var.get(), highlightthickness=1, highlightbackground="#ffffff")
        self.sample_preview.pack(side=tk.RIGHT, padx=(8, 0))
        self.sample_preview.pack_propagate(False)
        
        sample_color_btn = ttk.Button(sample_color_frame, text="Pick Color", style="TButton", width=11, command=lambda: self.pick_color(self.sample_color_var, self.sample_preview))
        sample_color_btn.pack(side=tk.RIGHT)

        run_btn = ttk.Button(self.control_frame, text="Generate Visualization", command=self.update_plots)
        run_btn.pack(fill=tk.X, ipady=12, pady=(15, 0))

        self.plot_frame = ttk.Frame(self.root, padding=15, style="TFrame")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, self.axes = plt.subplots(1, 2, figsize=(9, 5), facecolor=self.COLOR_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        bottom_bar_layout = tk.Frame(self.plot_frame, bg=self.COLOR_BG)
        bottom_bar_layout.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

        self.toolbar = NavigationToolbar2Tk(self.canvas, bottom_bar_layout)
        self.toolbar.configure(background=self.COLOR_BG)
        
        self.root.update_idletasks()

        home_btn = self.toolbar.children.get('!button')
        native_back = self.toolbar.children.get('!button2')
        native_forward = self.toolbar.children.get('!button3')

        if native_back:
            native_back.pack_forget()
        if native_forward:
            native_forward.pack_forget()

        for widget_name, widget in self.toolbar.children.items():
            c_class = widget.winfo_class()
            if c_class in ('Button', 'Checkbutton'):
                widget.configure(
                    background=self.COLOR_BG,
                    foreground="black",
                    disabledforeground="#4b5263",
                    activebackground=self.COLOR_PANEL,
                    activeforeground="black",
                    bd=0,
                    highlightthickness=0,
                    relief="flat"
                )
                
                if c_class == 'Checkbutton':
                    try:
                        widget.configure(selectcolor=self.COLOR_PANEL)
                    except Exception:
                        pass
                
                if widget_name not in ('!button2', '!button3'):
                    widget.pack_configure(padx=3, pady=2, ipadx=4, ipady=2, side=tk.LEFT)
                    
            elif c_class == 'Label':
                widget.configure(fg="#ffffff", font=("Segoe UI", 9, "bold"), bg=self.COLOR_BG)

        custom_back_btn = tk.Button(self.toolbar, text="◀", font=("Segoe UI", 12, "bold"),
                                    bg=self.COLOR_BG, fg="black", activebackground=self.COLOR_PANEL,
                                    activeforeground="black", bd=0, highlightthickness=0, width=3,
                                    command=self.toolbar.back)
        
        custom_forward_btn = tk.Button(self.toolbar, text="▶", font=("Segoe UI", 12, "bold"),
                                       bg=self.COLOR_BG, fg="black", activebackground=self.COLOR_PANEL,
                                       activeforeground="black", bd=0, highlightthickness=0, width=3,
                                       command=self.toolbar.forward)

        if hasattr(self.toolbar, 'message'):
            custom_back_btn.bind("<Enter>", lambda e: self.toolbar.message.set("Back to previous view"))
            custom_back_btn.bind("<Leave>", lambda e: self.toolbar.message.set(""))
            custom_forward_btn.bind("<Enter>", lambda e: self.toolbar.message.set("Forward to next view"))
            custom_forward_btn.bind("<Leave>", lambda e: self.toolbar.message.set(""))
            
            tooltip_map = {
                '!button': "Reset original view",
                '!checkbutton': "Pan axes with left mouse, zoom with right",
                '!checkbutton2': "Zoom to rectangle",
                '!button4': "Configure subplots",
                '!button5': "Save the figure"
            }
            
            for widget_name, widget in self.toolbar.children.items():
                if widget_name in tooltip_map:
                    msg = tooltip_map[widget_name]
                    widget.bind("<Enter>", lambda e, m=msg: self.toolbar.message.set(m), add="+")
                    widget.bind("<Leave>", lambda e: self.toolbar.message.set(""), add="+")

        if home_btn:
            custom_back_btn.pack(side=tk.LEFT, padx=3, pady=2, ipadx=4, ipady=2, after=home_btn)
            custom_forward_btn.pack(side=tk.LEFT, padx=3, pady=2, ipadx=4, ipady=2, after=custom_back_btn)

        if hasattr(self.toolbar, 'message'):
            self.toolbar.message.set("") 
            for key in self.toolbar.children:
                if 'label' in key:
                    self.toolbar.children[key].configure(fg="#ffffff", font=("Segoe UI", 9, "bold"), bg=self.COLOR_BG)

        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

        self.update_plots()
        self.toggle_btn.configure(command=self.toggle_sidebar)

    def update_plots(self):
        dist = self.dist_var.get()
        n = int(self.n_var.get())
        trials = int(self.trials_var.get())
        
        self.axes[0].clear()
        self.axes[1].clear()
        
        for ax in self.axes:
            ax.set_facecolor(self.COLOR_PANEL)
            ax.tick_params(colors=self.COLOR_TEXT, labelsize=9)
            ax.spines['bottom'].set_color(self.COLOR_PANEL)
            ax.spines['top'].set_color(self.COLOR_PANEL)
            ax.spines['left'].set_color(self.COLOR_PANEL)
            ax.spines['right'].set_color(self.COLOR_PANEL)
        
        if dist == "uniform":
            population = np.random.uniform(0, 1, 100000)
        elif dist == "exponential":
            population = np.random.exponential(scale=1, size=100000)
            
        samples = np.random.choice(population, size=(trials, n))
        sample_means = np.mean(samples, axis=1)
        
        self.axes[0].hist(population[:1000], bins=40, color=self.pop_color_var.get(), edgecolor=self.COLOR_PANEL, alpha=0.85)
        self.axes[0].set_title(f"Population Profile ({dist.capitalize()})", fontsize=11, fontweight='bold', color=self.COLOR_HEADER, pad=12)
        self.axes[0].grid(axis='y', color=self.COLOR_BG, linestyle=':', alpha=0.5)
        
        self.axes[1].hist(sample_means, bins=50, color=self.sample_color_var.get(), edgecolor=self.COLOR_PANEL, density=True, label="Sample Means", alpha=0.85)
        self.axes[1].set_title(f"Sample Means Distribution (n={n})", fontsize=11, fontweight='bold', color=self.COLOR_HEADER, pad=12)
        self.axes[1].grid(axis='y', color=self.COLOR_BG, linestyle=':', alpha=0.5)
        
        mu, sigma = np.mean(sample_means), np.std(sample_means)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        
        self.axes[1].plot(x, y, color="#ffffff", lw=2, label="Normal Curve Profile")
        
        leg = self.axes[1].legend(frameon=True, facecolor=self.COLOR_BG, edgecolor="none", loc="upper right")
        for text in leg.get_texts():
            text.set_color(self.COLOR_TEXT)
            text.set_size(8)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def create_linked_control(self, parent, label_text, variable):
        header_frame = ttk.Frame(parent, style="Panel.TFrame")
        header_frame.pack(fill=tk.X, pady=(12, 4))
        
        ttk.Label(header_frame, text=label_text, style="Sub.TLabel").pack(side=tk.LEFT)
        
        entry = tk.Entry(header_frame, textvariable=variable, width=6, justify="center",
                         bg="#1e222b", fg="#ffffff", insertbackground="#ffffff",
                         disabledbackground="#1e222b", disabledforeground="#ffffff",
                         font=("Segoe UI", 10, "bold"), bd=0, highlightthickness=1, 
                         highlightbackground="#3e4451", highlightcolor=self.COLOR_ACCENT)
        entry.pack(side=tk.RIGHT, ipady=3)
        
        slider = tk.Scale(parent, variable=variable, orient=tk.HORIZONTAL, showvalue=False,
                          troughcolor=self.COLOR_BG, highlightthickness=0, bd=0, cursor="hand2",
                          bg="#ffffff",
                          activebackground="#ffffff",
                          fg=self.COLOR_TEXT)
        slider.pack(fill=tk.X, pady=(0, 10))
        
        return slider
    
    def validate_entry(self, variable, min_val, max_val):
        try:
            val = variable.get()
            if val < min_val: variable.set(min_val)
            elif val > max_val: variable.set(max_val)
        except tk.TclError:
            pass

    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.side_container.pack_forget()
            self.toggle_btn.configure(text="▶")
            self.sidebar_visible = False
        else:
            self.side_container.pack(side=tk.LEFT, fill=tk.Y, before=self.toggle_btn)
            self.toggle_btn.configure(text="◀")
            self.sidebar_visible = True
        self.finalize_chart_layout()

    def finalize_chart_layout(self):
        self.fig.tight_layout()
        self.canvas.draw()

    def pick_color(self, hex_variable, preview_widget):
        selected_color = colorchooser.askcolor(initialcolor=hex_variable.get(), title="Select Histogram Color")
        if selected_color[1]:
            hex_variable.set(selected_color[1])
            preview_widget.configure(bg=selected_color[1])

if __name__ == "__main__":
    root = tk.Tk()
    app = CLTVisualizerApp(root)
    root.mainloop()
    