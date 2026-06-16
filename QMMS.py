import tkinter as tk
from tkinter import ttk
from tkinter import colorchooser
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# used the TkAgg backend to properly integrate matplotlib elements within a Tkinter window
mpl.rcParams['toolbar'] = 'toolbar2'
mpl.rcParams['backend'] = 'TkAgg'

class CLTVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Central Limit Theorem Visualizer")
        self.root.geometry("1180x740")
        
        # color palette definition
        self.COLOR_BG = "#1e222b" 
        self.COLOR_PANEL = "#282c34"
        self.COLOR_TEXT = "#abb2bf"
        self.COLOR_HEADER = "#ffffff"
        self.COLOR_ACCENT = "#4b6cb7"
        self.COLOR_HOVER = "#5c7ed8"
        
        self.root.configure(bg=self.COLOR_BG)

        # combobox (dropdown) global styling
        root.option_add("*TCombobox*Listbox.background", self.COLOR_PANEL)
        root.option_add("*TCombobox*Listbox.foreground", "#ffffff")
        root.option_add("*TCombobox*Listbox.selectBackground", self.COLOR_ACCENT)
        root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))
        
        # application style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam") # 'clam' theme allows for deeper color customization
        
        # general frame styles
        self.style.configure("TFrame", background=self.COLOR_BG)
        self.style.configure("Panel.TFrame", background=self.COLOR_PANEL)
        
        # label styles for hierarchy 
        self.style.configure("TLabel", background=self.COLOR_PANEL, foreground=self.COLOR_TEXT, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), background=self.COLOR_PANEL, foreground=self.COLOR_HEADER)
        self.style.configure("Sub.TLabel", font=("Segoe UI", 10, "bold"), background=self.COLOR_PANEL, foreground=self.COLOR_HEADER)
        
        # primary action button style
        self.style.configure("TButton", 
                             font=("Segoe UI", 10, "bold"), 
                             background=self.COLOR_ACCENT, 
                             foreground="white", 
                             borderwidth=0,
                             focusthickness=0,
                             relief="flat",
                             padding=(10, 6))
        self.style.map("TButton", background=[("active", self.COLOR_HOVER)])

        # sidebar toggle button style
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

        # combobox specific style
        self.style.configure("TCombobox", 
                             fieldbackground="#ffffff",
                             background=self.COLOR_PANEL,
                             foreground="#1e222b",
                             borderwidth=1, 
                             arrowcolor=self.COLOR_TEXT)
        self.style.map("TCombobox", 
                       fieldbackground=[("readonly", "#ffffff")],
                       foreground=[("readonly", "#1e222b")])

        # sidebar layout setup
        self.sidebar_visible = True
        self.sidebar_width = 300
        self.is_animating = False

        # main container for the sidebar components
        self.side_container = ttk.Frame(self.root, style="TFrame")
        self.side_container.pack(side=tk.LEFT, fill=tk.Y)

        # panel holding the actual controls
        self.control_frame = ttk.Frame(self.side_container, padding=25, style="Panel.TFrame", width=self.sidebar_width)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.control_frame.pack_propagate(False) # prevents inner widgets from expanding the set width

        # collapse/Expand Button
        self.toggle_btn = ttk.Button(self.root, text="◀", width=3, style="Toggle.TButton")
        self.toggle_btn.pack(side=tk.LEFT, fill=tk.Y)
        
        # simulation controls section
        ttk.Label(self.control_frame, text="Simulation Controls", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 15))
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 15))
        
        # base population distribution selection
        ttk.Label(self.control_frame, text="Population Distribution:", style="Sub.TLabel").pack(anchor=tk.W, pady=(5, 5))
        self.dist_var = tk.StringVar(value="exponential")
        dist_cb = ttk.Combobox(self.control_frame, textvariable=self.dist_var, values=["exponential", "uniform"], state="readonly")
        dist_cb.pack(fill=tk.X, pady=(0, 15))

        # tkinter variables for sliders (n for sample Size, trials for iterations)
        self.n_var = tk.IntVar(value=30)
        self.trials_var = tk.IntVar(value=2000)
        
        # real-time input validation to constrain values
        self.n_var.trace_add("write", lambda *args: self.validate_entry(self.n_var, 2, 100))
        self.trials_var.trace_add("write", lambda *args: self.validate_entry(self.trials_var, 100, 10000))
        
        # custom linked controls it links a label, text entry, and slider to one variable
        self.n_slider = self.create_linked_control(self.control_frame, "Sample Size (n):", self.n_var)
        self.n_slider.configure(from_=2, to=100)

        self.trials_slider = self.create_linked_control(self.control_frame, "Number of Trials:", self.trials_var)
        self.trials_slider.configure(from_=100, to=10000, resolution=100)

        # graph customization section
        ttk.Label(self.control_frame, text="Graph Customization", style="Header.TLabel").pack(anchor=tk.W, pady=(20, 15))
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 15))
        
        # default bar chart colors
        self.pop_color_var = tk.StringVar(value="#ff7f50") # Coral
        self.sample_color_var = tk.StringVar(value="#4682b4") # Steel Blue

        # population color picker UI
        pop_color_frame = ttk.Frame(self.control_frame, style="Panel.TFrame")
        pop_color_frame.pack(fill=tk.X, pady=(5, 10))
        ttk.Label(pop_color_frame, text="Population Color:").pack(side=tk.LEFT)
        
        # visual color preview box
        self.pop_preview = tk.Frame(pop_color_frame, width=22, height=22, bg=self.pop_color_var.get(), highlightthickness=1, highlightbackground="#ffffff")
        self.pop_preview.pack(side=tk.RIGHT, padx=(8, 0))
        self.pop_preview.pack_propagate(False)
        
        pop_color_btn = ttk.Button(pop_color_frame, text="Pick Color", style="TButton", width=11, command=lambda: self.pick_color(self.pop_color_var, self.pop_preview))
        pop_color_btn.pack(side=tk.RIGHT)

        # sample color picker UI
        sample_color_frame = ttk.Frame(self.control_frame, style="Panel.TFrame")
        sample_color_frame.pack(fill=tk.X, pady=(5, 25))
        ttk.Label(sample_color_frame, text="Sample Color:").pack(side=tk.LEFT)
        
        # visual color preview box
        self.sample_preview = tk.Frame(sample_color_frame, width=22, height=22, bg=self.sample_color_var.get(), highlightthickness=1, highlightbackground="#ffffff")
        self.sample_preview.pack(side=tk.RIGHT, padx=(8, 0))
        self.sample_preview.pack_propagate(False)
        
        sample_color_btn = ttk.Button(sample_color_frame, text="Pick Color", style="TButton", width=11, command=lambda: self.pick_color(self.sample_color_var, self.sample_preview))
        sample_color_btn.pack(side=tk.RIGHT)

        # primary trigger button (recalculates data)
        run_btn = ttk.Button(self.control_frame, text="Generate Visualization", command=self.update_plots)
        run_btn.pack(fill=tk.X, ipady=12, pady=(15, 0))

        # matplotlib canvas initialization
        self.plot_frame = ttk.Frame(self.root, padding=15, style="TFrame")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # create a matplotlib figure (1 row, 2 columns for the two histograms)
        self.fig, self.axes = plt.subplots(1, 2, figsize=(9, 5), facecolor=self.COLOR_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # bottom frame dedicated to the matplotlib toolbar
        bottom_bar_layout = tk.Frame(self.plot_frame, bg=self.COLOR_BG)
        bottom_bar_layout.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

        # customizing the default matplotlib navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, bottom_bar_layout)
        self.toolbar.configure(background=self.COLOR_BG)
        
        self.root.update_idletasks() # refresh to let Matplotlib render its widgets before we hack them

        # isolate matplotlib's native buttons so we can override/style them
        home_btn = self.toolbar.children.get('!button')
        native_back = self.toolbar.children.get('!button2')
        native_forward = self.toolbar.children.get('!button3')

        # hide native back/forward buttons (we'll replace them below)
        if native_back: native_back.pack_forget()
        if native_forward: native_forward.pack_forget()

        # restyle remaining native widgets to match the dark theme
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
                
                # checkbuttons (pan, zoom) require specific styling
                if c_class == 'Checkbutton':
                    try: widget.configure(selectcolor=self.COLOR_PANEL)
                    except Exception: pass
                
                # add padding to remaining buttons
                if widget_name not in ('!button2', '!button3'):
                    widget.pack_configure(padx=3, pady=2, ipadx=4, ipady=2, side=tk.LEFT)
                    
            # restyle coordinates label
            elif c_class == 'Label':
                widget.configure(fg="#ffffff", font=("Segoe UI", 9, "bold"), bg=self.COLOR_BG)

        # generate Custom replacement back and forward buttons
        custom_back_btn = tk.Button(self.toolbar, text="◀", font=("Segoe UI", 12, "bold"),
                                    bg=self.COLOR_BG, fg="black", activebackground=self.COLOR_PANEL,
                                    activeforeground="black", bd=0, highlightthickness=0, width=3,
                                    command=self.toolbar.back)
        
        custom_forward_btn = tk.Button(self.toolbar, text="▶", font=("Segoe UI", 12, "bold"),
                                       bg=self.COLOR_BG, fg="black", activebackground=self.COLOR_PANEL,
                                       activeforeground="black", bd=0, highlightthickness=0, width=3,
                                       command=self.toolbar.forward)

        # bind hover tooltips for navigation accessibility
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

        # inject custom buttons into layout right after the home icon
        if home_btn:
            custom_back_btn.pack(side=tk.LEFT, padx=3, pady=2, ipadx=4, ipady=2, after=home_btn)
            custom_forward_btn.pack(side=tk.LEFT, padx=3, pady=2, ipadx=4, ipady=2, after=custom_back_btn)

        # ensure the coordinate tracker label correctly renders text colors
        if hasattr(self.toolbar, 'message'):
            self.toolbar.message.set("") 
            for key in self.toolbar.children:
                if 'label' in key:
                    self.toolbar.children[key].configure(fg="#ffffff", font=("Segoe UI", 9, "bold"), bg=self.COLOR_BG)

        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

        # initial plot render
        self.update_plots()
        
        # link sidebar toggle to callback
        self.toggle_btn.configure(command=self.toggle_sidebar)

    def update_plots(self):
        """Generates random data according to controls, runs the math, and renders graphs."""
        dist = self.dist_var.get()
        n = int(self.n_var.get())
        trials = int(self.trials_var.get())
        
        # wipe old plots clean
        self.axes[0].clear()
        self.axes[1].clear()
        
        # apply dark mode styles to plot bounds and ticks
        for ax in self.axes:
            ax.set_facecolor(self.COLOR_PANEL)
            ax.tick_params(colors=self.COLOR_TEXT, labelsize=9)
            ax.spines['bottom'].set_color(self.COLOR_PANEL)
            ax.spines['top'].set_color(self.COLOR_PANEL)
            ax.spines['left'].set_color(self.COLOR_PANEL)
            ax.spines['right'].set_color(self.COLOR_PANEL)
        
        # first step establish initial population
        # we generate a massive array of 100,000 values to act as our core population
        if dist == "uniform":
            population = np.random.uniform(0, 1, 100000)
        elif dist == "exponential":
            population = np.random.exponential(scale=1, size=100000)
            
        # 2nd step sampling process
        # using numpy to grab randomized selections matrix-style `trials` rows and `n` columns
        samples = np.random.choice(population, size=(trials, n))
        
        # calculate the mean or average of each individual sample array
        sample_means = np.mean(samples, axis=1)
        
        # graph no 1 or the raw population base
        self.axes[0].hist(population[:1000], bins=40, color=self.pop_color_var.get(), edgecolor=self.COLOR_PANEL, alpha=0.85)
        self.axes[0].set_title(f"Population Profile ({dist.capitalize()})", fontsize=11, fontweight='bold', color=self.COLOR_HEADER, pad=12)
        self.axes[0].grid(axis='y', color=self.COLOR_BG, linestyle=':', alpha=0.5)
        
        # graph no 2 or the distribution of sample means
        # used density=True to normalize the height.
        self.axes[1].hist(sample_means, bins=50, color=self.sample_color_var.get(), edgecolor=self.COLOR_PANEL, density=True, label="Sample Means", alpha=0.85)
        self.axes[1].set_title(f"Sample Means Distribution (n={n})", fontsize=11, fontweight='bold', color=self.COLOR_HEADER, pad=12)
        self.axes[1].grid(axis='y', color=self.COLOR_BG, linestyle=':', alpha=0.5)
        
        # 3rd step overlay the theoretical normal curve
        # Calculate expected mu (mean) and sigma (std deviation)
        mu, sigma = np.mean(sample_means), np.std(sample_means)
        
        # create an X-axis spanning exactly 4 standard deviations left and right
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        
        # apply the probability density function formula for normal distribution to build Y coordinates
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        
        self.axes[1].plot(x, y, color="#ffffff", lw=2, label="Normal Curve Profile")
        
        # setup aesthetic legend
        leg = self.axes[1].legend(frameon=True, facecolor=self.COLOR_BG, edgecolor="none", loc="upper right")
        for text in leg.get_texts():
            text.set_color(self.COLOR_TEXT)
            text.set_size(8)
        
        # apply strict fit and execute drawing command
        self.fig.tight_layout()
        self.canvas.draw()

    def create_linked_control(self, parent, label_text, variable):
        """Builds a UI set mapping a slider and a typed entry box to the same Tkinter Variable."""
        header_frame = ttk.Frame(parent, style="Panel.TFrame")
        header_frame.pack(fill=tk.X, pady=(12, 4))
        
        ttk.Label(header_frame, text=label_text, style="Sub.TLabel").pack(side=tk.LEFT)
        
        # type-in entry widget
        entry = tk.Entry(header_frame, textvariable=variable, width=6, justify="center",
                         bg="#1e222b", fg="#ffffff", insertbackground="#ffffff",
                         disabledbackground="#1e222b", disabledforeground="#ffffff",
                         font=("Segoe UI", 10, "bold"), bd=0, highlightthickness=1, 
                         highlightbackground="#3e4451", highlightcolor=self.COLOR_ACCENT)
        entry.pack(side=tk.RIGHT, ipady=3)
        
        # slider widget
        slider = tk.Scale(parent, variable=variable, orient=tk.HORIZONTAL, showvalue=False,
                          troughcolor=self.COLOR_BG, highlightthickness=0, bd=0, cursor="hand2",
                          bg="#ffffff",
                          activebackground="#ffffff",
                          fg=self.COLOR_TEXT)
        slider.pack(fill=tk.X, pady=(0, 10))
        
        return slider
    
    def validate_entry(self, variable, min_val, max_val):
        """Forces integers linked to an Entry Widget to remain within set numeric bounds."""
        try:
            val = variable.get()
            if val < min_val: variable.set(min_val)
            elif val > max_val: variable.set(max_val)
        except tk.TclError:
            pass # fails gracefully if the user briefly deletes the entire number in the box

    def toggle_sidebar(self):
        """Hides or reveals the control menu on the left side."""
        if self.sidebar_visible:
            self.side_container.pack_forget()
            self.toggle_btn.configure(text="▶")
            self.sidebar_visible = False
        else:
            # re-insert the sidebar right before the toggle button
            self.side_container.pack(side=tk.LEFT, fill=tk.Y, before=self.toggle_btn)
            self.toggle_btn.configure(text="◀")
            self.sidebar_visible = True
            
        self.finalize_chart_layout()

    def finalize_chart_layout(self):
        """Refreshes rendering boundaries so the graphs resize instantly when the sidebar closes/opens."""
        self.fig.tight_layout()
        self.canvas.draw()

    def pick_color(self, hex_variable, preview_widget):
        """Spawns an OS color dialog, grabbing the resulting HEX to update our plot styles."""
        selected_color = colorchooser.askcolor(initialcolor=hex_variable.get(), title="Select Histogram Color")
        if selected_color[1]: # returns (RGB_Tuple, Hex_String). checks if user didn't hit cancel
            hex_variable.set(selected_color[1])
            preview_widget.configure(bg=selected_color[1])

# application boot block :>
if __name__ == "__main__":
    root = tk.Tk()
    app = CLTVisualizerApp(root)
    root.mainloop()