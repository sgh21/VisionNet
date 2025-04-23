#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import time
from threading import Thread
import queue

class TerraceImageSmoother:
    def __init__(self, image):
        """Initialize smoother"""
        self.image = image
        
    def detect_edges(self, threshold=1):
        """
        Detect edges in image (value change areas)
        
        Args:
            threshold: Edge detection threshold, adjacent pixel differences 
                      greater than this value are considered edges
            
        Returns:
            Edge mask (binary image)
        """
        # Calculate gradients in horizontal and vertical directions
        grad_x = np.abs(np.diff(self.image, axis=1, prepend=self.image[:, :1]))
        grad_y = np.abs(np.diff(self.image, axis=0, prepend=self.image[:1, :]))
        
        # Combine gradients
        gradient = np.maximum(grad_x, grad_y)
        
        # Threshold to create edge mask
        edge_mask = gradient > threshold
        
        return edge_mask
    
    def generate_distance_map(self, edge_mask, max_distance):
        """
        Generate distance map to nearest edge
        
        Args:
            edge_mask: Edge mask
            max_distance: Maximum considered distance
            
        Returns:
            Distance map to nearest edge
        """
        # Use OpenCV's distance transform to get distance to nearest edge
        dist_transform = cv2.distanceTransform(
            (1 - edge_mask).astype(np.uint8), 
            cv2.DIST_L2, 
            cv2.DIST_MASK_PRECISE
        )
        
        # Truncate to maximum distance
        dist_transform = np.minimum(dist_transform, max_distance)
        
        return dist_transform
    
    def smooth_terrace_image(self, transition_width=5, smoothness=2.0, preserve_edges=0.5, threshold=1):
        """
        Smooth terrace-like image edges
        
        Args:
            transition_width: Transition width (pixels)
            smoothness: Smoothing degree (1.0-10.0), higher is smoother
            preserve_edges: Edge preservation (0.0-1.0), higher preserves more
            threshold: Edge detection threshold
            
        Returns:
            Smoothed image, edge mask, distance map, weight map
        """
        # 1. Detect edges
        edge_mask = self.detect_edges(threshold)
        
        # 2. Calculate distance to nearest edge
        dist_map = self.generate_distance_map(edge_mask, transition_width)
        
        # 3. Create smoothing weight map
        # Use sigmoid function for smooth transitions
        weight = 1.0 / (1.0 + np.exp(-(dist_map - transition_width/2) * smoothness / transition_width))
        
        # 4. Create smoothed image
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(self.image, (2*transition_width+1, 2*transition_width+1), 0)
        
        # 5. Blend original and blurred images based on distance and edge preservation
        edge_weight = weight * (1.0 - preserve_edges) + preserve_edges
        
        # Weighted blend of original and smoothed images
        smoothed = self.image * edge_weight + blurred * (1.0 - edge_weight)
        
        return smoothed, edge_mask, dist_map, weight

def generate_test_image(size=560, levels=5, noise=0.0):
    """
    Generate test terrace-like image
    
    Args:
        size: Image size
        levels: Number of terrace levels
        noise: Amount of noise to add
        
    Returns:
        Generated test image
    """
    # Create basic gradient
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Create radial gradient
    gradient = np.sqrt((x - 0.5)**2 + (y - 0.5)**2) * 2
    
    # Quantize to specified number of levels
    terrace = np.floor(gradient * levels) / levels
    
    # Optional: add some noise
    if noise > 0:
        terrace += np.random.normal(0, noise, terrace.shape)
    
    # Normalize to 0-255 range
    terrace = (terrace * 255).astype(np.uint8)
    
    return terrace

class TerraceSmootherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Terrace Image Edge Smoothing Tool")
        self.root.geometry("1280x800")
        self.root.configure(bg="#f0f0f0")
        
        self.image = None
        self.smoothed_image = None
        self.edge_mask = None
        self.dist_map = None
        self.weight_map = None
        
        # Processing queue and flag
        self.process_queue = queue.Queue()
        self.is_processing = False
        
        # Create UI
        self.create_ui()
        
        # Generate test image if none loaded
        if self.image is None:
            self.generate_test_image()
        
        # Start processing thread
        self.start_process_thread()
        
    def create_ui(self):
        """Create user interface"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # File operation buttons
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(file_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Generate Test Image", command=self.generate_test_image_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=5)
        
        # Parameter controls
        params_frame = ttk.LabelFrame(control_frame, text="Smoothing Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Transition width slider
        ttk.Label(params_frame, text="Transition Width:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.width_var = tk.IntVar(value=5)
        width_slider = ttk.Scale(params_frame, from_=1, to=20, variable=self.width_var, 
                                command=lambda _: self.update_param_label('width'))
        width_slider.grid(row=0, column=1, sticky=tk.EW, pady=5)
        self.width_label = ttk.Label(params_frame, text="5 pixels")
        self.width_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Smoothness slider
        ttk.Label(params_frame, text="Smoothness:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.smoothness_var = tk.DoubleVar(value=2.0)
        smoothness_slider = ttk.Scale(params_frame, from_=0.5, to=10.0, variable=self.smoothness_var,
                                     command=lambda _: self.update_param_label('smoothness'))
        smoothness_slider.grid(row=1, column=1, sticky=tk.EW, pady=5)
        self.smoothness_label = ttk.Label(params_frame, text="2.0")
        self.smoothness_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Edge preservation slider
        ttk.Label(params_frame, text="Edge Preservation:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.preserve_var = tk.DoubleVar(value=0.5)
        preserve_slider = ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.preserve_var,
                                   command=lambda _: self.update_param_label('preserve'))
        preserve_slider.grid(row=2, column=1, sticky=tk.EW, pady=5)
        self.preserve_label = ttk.Label(params_frame, text="0.5")
        self.preserve_label.grid(row=2, column=2, padx=5, pady=5)
        
        # Edge threshold slider
        ttk.Label(params_frame, text="Edge Threshold:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.threshold_var = tk.DoubleVar(value=1.0)
        threshold_slider = ttk.Scale(params_frame, from_=0.1, to=10.0, variable=self.threshold_var,
                                    command=lambda _: self.update_param_label('threshold'))
        threshold_slider.grid(row=3, column=1, sticky=tk.EW, pady=5)
        self.threshold_label = ttk.Label(params_frame, text="1.0")
        self.threshold_label.grid(row=3, column=2, padx=5, pady=5)
        
        # Apply button
        ttk.Button(params_frame, text="Apply", command=self.process_image).grid(row=4, column=0, columnspan=3, pady=10)
        
        # Auto-refresh checkbox
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Auto-refresh when parameters change", variable=self.auto_refresh_var).grid(
            row=5, column=0, columnspan=3, pady=5)
        
        # Display settings
        view_frame = ttk.LabelFrame(control_frame, text="Display Settings", padding=10)
        view_frame.pack(fill=tk.X, pady=(0, 15))
        
        # View mode selection
        ttk.Label(view_frame, text="View Mode:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.view_mode_var = tk.StringVar(value="2D")
        ttk.Radiobutton(view_frame, text="2D View", variable=self.view_mode_var, value="2D", 
                       command=self.update_view).grid(row=0, column=1, pady=5)
        ttk.Radiobutton(view_frame, text="3D View", variable=self.view_mode_var, value="3D", 
                       command=self.update_view).grid(row=0, column=2, pady=5)
        
        # Image view selection
        ttk.Label(view_frame, text="Image Selection:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.view_image_var = tk.StringVar(value="combined")
        view_image_combo = ttk.Combobox(view_frame, textvariable=self.view_image_var,
                                      values=["combined", "original", "smoothed", "difference", 
                                             "edges", "distance", "weights"])
        view_image_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=5)
        view_image_combo.bind("<<ComboboxSelected>>", lambda _: self.update_view())
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        
        # Image display area 
        self.figure_frame = ttk.Frame(main_frame)
        self.figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create Matplotlib figure
        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bind slider release events
        width_slider.bind("<ButtonRelease-1>", lambda _: self.on_slider_release())
        smoothness_slider.bind("<ButtonRelease-1>", lambda _: self.on_slider_release())
        preserve_slider.bind("<ButtonRelease-1>", lambda _: self.on_slider_release())
        threshold_slider.bind("<ButtonRelease-1>", lambda _: self.on_slider_release())
        
    def update_param_label(self, param_name):
        """Update parameter label display"""
        if param_name == 'width':
            value = self.width_var.get()
            self.width_label.config(text=f"{value} pixels")
        elif param_name == 'smoothness':
            value = self.smoothness_var.get()
            self.smoothness_label.config(text=f"{value:.1f}")
        elif param_name == 'preserve':
            value = self.preserve_var.get()
            self.preserve_label.config(text=f"{value:.2f}")
        elif param_name == 'threshold':
            value = self.threshold_var.get()
            self.threshold_label.config(text=f"{value:.1f}")
    
    def on_slider_release(self):
        """Callback when slider is released"""
        if self.auto_refresh_var.get():
            self.process_image()
    
    def start_process_thread(self):
        """Start processing thread"""
        self.process_thread = Thread(target=self.process_worker, daemon=True)
        self.process_thread.start()
    
    def process_worker(self):
        """Processing thread worker function"""
        while True:
            try:
                # Wait for task
                task = self.process_queue.get()
                
                # Set processing flag
                self.is_processing = True
                self.root.after(0, lambda: self.status_var.set("Processing..."))
                
                # Skip if no original image
                if self.image is None:
                    self.is_processing = False
                    self.process_queue.task_done()
                    continue
                
                # Get parameters
                width = self.width_var.get()
                smoothness = self.smoothness_var.get()
                preserve = self.preserve_var.get()
                threshold = self.threshold_var.get()
                
                # Create smoother
                smoother = TerraceImageSmoother(self.image)
                
                # Apply smoothing
                smoothed, edge_mask, dist_map, weight_map = smoother.smooth_terrace_image(
                    width, smoothness, preserve, threshold
                )
                
                # Update images
                self.smoothed_image = smoothed
                self.edge_mask = edge_mask
                self.dist_map = dist_map
                self.weight_map = weight_map
                
                # Update display
                self.root.after(0, self.update_view)
                
                # Update status
                self.root.after(0, lambda: self.status_var.set("Processing complete"))
                
                # Clear processing flag
                self.is_processing = False
                
                # Mark task as done
                self.process_queue.task_done()
            except Exception as e:
                print(f"Processing error: {e}")
                self.root.after(0, lambda: self.status_var.set(f"Processing error: {e}"))
                self.is_processing = False
                self.process_queue.task_done()
    
    def process_image(self):
        """Process image"""
        if not self.is_processing:
            self.process_queue.put(True)
    
    def open_image(self):
        """Open image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                # Load image
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("Cannot read image")
                
                # Resize if image is too large
                max_size = 800
                h, w = img.shape
                if h > max_size or w > max_size:
                    scale = max_size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    self.status_var.set(f"Image loaded and resized to {new_w}x{new_h}")
                else:
                    self.status_var.set(f"Image loaded: {w}x{h}")
                
                # Update image
                self.image = img.astype(np.float32)
                
                # Process image
                self.process_image()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {e}")
                self.status_var.set("Failed to load image")
    
    def generate_test_image_dialog(self):
        """Generate test image dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Generate Test Image")
        dialog.geometry("300x200")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Set up dialog content
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Size setting
        ttk.Label(frame, text="Image Size:").grid(row=0, column=0, sticky=tk.W, pady=5)
        size_var = tk.IntVar(value=560)
        ttk.Spinbox(frame, from_=100, to=1000, textvariable=size_var, width=10).grid(row=0, column=1, pady=5)
        
        # Levels setting
        ttk.Label(frame, text="Terrace Levels:").grid(row=1, column=0, sticky=tk.W, pady=5)
        levels_var = tk.IntVar(value=5)
        ttk.Spinbox(frame, from_=2, to=20, textvariable=levels_var, width=10).grid(row=1, column=1, pady=5)
        
        # Noise setting
        ttk.Label(frame, text="Noise Level:").grid(row=2, column=0, sticky=tk.W, pady=5)
        noise_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(frame, from_=0.0, to=0.1, increment=0.01, textvariable=noise_var, width=10).grid(row=2, column=1, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        def on_generate():
            try:
                size = size_var.get()
                levels = levels_var.get()
                noise = noise_var.get()
                
                # Generate image
                self.image = generate_test_image(size, levels, noise)
                
                # Update status
                self.status_var.set(f"Test image generated: {size}x{size}, {levels} levels, noise={noise:.2f}")
                
                # Process image
                self.process_image()
                
                # Close dialog
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error generating image: {e}")
        
        ttk.Button(button_frame, text="Generate", command=on_generate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def generate_test_image(self):
        """Generate default test image"""
        self.image = generate_test_image(560, 5, 0.0)
        self.status_var.set("Default test image generated: 560x560, 5 levels, no noise")
        self.process_image()
    
    def update_view(self):
        """Update graphical display"""
        if self.image is None:
            return
            
        # Clear current figure
        self.fig.clear()
        
        # Get display mode
        view_mode = self.view_mode_var.get()
        view_image = self.view_image_var.get()
        
        if view_mode == "2D":
            self.update_2d_view(view_image)
        else:
            self.update_3d_view(view_image)
        
        # Refresh canvas
        self.canvas.draw()
    
    def update_2d_view(self, view_image):
        """Update 2D view"""
        if view_image == "combined":
            # Combined view mode - show multiple images
            
            # Create 2x3 layout
            gs = self.fig.add_gridspec(2, 3)
            
            # Original image
            ax1 = self.fig.add_subplot(gs[0, 0])
            ax1.imshow(self.image, cmap='viridis')
            ax1.set_title("Original Image")
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Smoothed image
            if self.smoothed_image is not None:
                ax2 = self.fig.add_subplot(gs[0, 1])
                ax2.imshow(self.smoothed_image, cmap='viridis')
                ax2.set_title("Smoothed Image")
                ax2.set_xticks([])
                ax2.set_yticks([])
            
            # Difference image
            if self.smoothed_image is not None:
                ax3 = self.fig.add_subplot(gs[0, 2])
                diff = self.smoothed_image - self.image
                vmax = max(abs(diff.min()), abs(diff.max()))
                ax3.imshow(diff, cmap='seismic', vmin=-vmax, vmax=vmax)
                ax3.set_title("Difference")
                ax3.set_xticks([])
                ax3.set_yticks([])
            
            # Edge image
            if self.edge_mask is not None:
                ax4 = self.fig.add_subplot(gs[1, 0])
                ax4.imshow(self.edge_mask, cmap='gray')
                ax4.set_title("Detected Edges")
                ax4.set_xticks([])
                ax4.set_yticks([])
            
            # Distance map
            if self.dist_map is not None:
                ax5 = self.fig.add_subplot(gs[1, 1])
                ax5.imshow(self.dist_map, cmap='hot')
                ax5.set_title("Distance to Edges")
                ax5.set_xticks([])
                ax5.set_yticks([])
            
            # Weight map
            if self.weight_map is not None:
                ax6 = self.fig.add_subplot(gs[1, 2])
                ax6.imshow(self.weight_map, cmap='winter')
                ax6.set_title("Smoothing Weights")
                ax6.set_xticks([])
                ax6.set_yticks([])
            
        else:
            # Single view mode
            ax = self.fig.add_subplot(111)
            
            if view_image == "original":
                ax.imshow(self.image, cmap='viridis')
                ax.set_title("Original Image")
            elif view_image == "smoothed" and self.smoothed_image is not None:
                ax.imshow(self.smoothed_image, cmap='viridis')
                ax.set_title("Smoothed Image")
            elif view_image == "difference" and self.smoothed_image is not None:
                diff = self.smoothed_image - self.image
                vmax = max(abs(diff.min()), abs(diff.max()))
                ax.imshow(diff, cmap='seismic', vmin=-vmax, vmax=vmax)
                ax.set_title("Difference")
            elif view_image == "edges" and self.edge_mask is not None:
                ax.imshow(self.edge_mask, cmap='gray')
                ax.set_title("Detected Edges")
            elif view_image == "distance" and self.dist_map is not None:
                ax.imshow(self.dist_map, cmap='hot')
                ax.set_title("Distance to Edges")
            elif view_image == "weights" and self.weight_map is not None:
                ax.imshow(self.weight_map, cmap='winter')
                ax.set_title("Smoothing Weights")
            else:
                ax.imshow(self.image, cmap='viridis')
                ax.set_title("Original Image")
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        self.fig.tight_layout()
    
    def update_3d_view(self, view_image):
        """Update 3D view"""
        # Get coordinate grid
        h, w = self.image.shape
        x = np.arange(0, w)
        y = np.arange(0, h)
        X, Y = np.meshgrid(x, y)
        
        # Reduce grid resolution for performance
        downsample = 4  # Downsampling factor
        X_down = X[::downsample, ::downsample]
        Y_down = Y[::downsample, ::downsample]
        
        # Create 1x2 layout for before/after
        gs = self.fig.add_gridspec(1, 2)
        
        # 3D original image
        ax1 = self.fig.add_subplot(gs[0, 0], projection='3d')
        Z_down = self.image[::downsample, ::downsample]
        ax1.plot_surface(X_down, Y_down, Z_down, cmap=cm.viridis, 
                        linewidth=0, antialiased=True)
        ax1.set_title("Original Terrain")
        
        # 3D smoothed image
        if self.smoothed_image is not None:
            ax2 = self.fig.add_subplot(gs[0, 1], projection='3d')
            Z_smooth_down = self.smoothed_image[::downsample, ::downsample]
            ax2.plot_surface(X_down, Y_down, Z_smooth_down, cmap=cm.viridis, 
                            linewidth=0, antialiased=True)
            ax2.set_title("Smoothed Terrain")
        
        self.fig.tight_layout()
    
    def save_result(self):
        """Save processing result"""
        if self.smoothed_image is None:
            messagebox.showinfo("Info", "No result to save")
            return
            
        # Choose save path
        file_path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPG Image", "*.jpg"), ("TIFF Image", "*.tif"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Save smoothed image
            cv2.imwrite(file_path, self.smoothed_image)
            
            # Get filename and extension
            base, ext = os.path.splitext(file_path)
            
            # Save other visualization results
            if self.edge_mask is not None:
                cv2.imwrite(f"{base}_edges{ext}", self.edge_mask.astype(np.uint8) * 255)
            
            if self.dist_map is not None:
                # Normalize distance map to 0-255
                dist_norm = self.dist_map * (255.0 / self.dist_map.max())
                cv2.imwrite(f"{base}_distance{ext}", dist_norm.astype(np.uint8))
            
            if self.weight_map is not None:
                # Normalize weight map to 0-255
                weight_norm = self.weight_map * 255.0
                cv2.imwrite(f"{base}_weights{ext}", weight_norm.astype(np.uint8))
            
            # Save heatmap version
            plt.imsave(f"{base}_heatmap{ext}", self.smoothed_image, cmap='viridis')
            
            self.status_var.set(f"Results saved to: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving result: {e}")
            self.status_var.set("Failed to save result")

def main():
    """Main function"""
    root = tk.Tk()
    app = TerraceSmootherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()