import pygame
import numpy as np
import sys
import math
from neuron import Perceptron
from datasets import get_gate_data, get_all_gates

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
PLOT_SIZE = 400
PLOT_MARGIN = 50
FPS = 2  # Slow animation for better visualization

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)

class NeuralNetworkAnimator:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Neural Network Training Animation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Animation state
        self.current_gate = "AND"
        self.epoch = 0
        self.training_complete = False
        self.paused = False
        
        # Get data
        self.X, self.y = get_gate_data(self.current_gate)
        self.perceptron = Perceptron(n_inputs=2, lr=0.1, epochs=100)
        
        # Training history for visualization
        self.weight_history = []
        self.error_history = []
        self.decision_boundaries = []
        
        # Setup training
        self.setup_training()
        
    def setup_training(self):
        """Setup training for the current gate"""
        self.X, self.y = get_gate_data(self.current_gate)
        self.perceptron = Perceptron(n_inputs=2, lr=0.1, epochs=100)
        self.weight_history = []
        self.error_history = []
        self.decision_boundaries = []
        self.epoch = 0
        self.training_complete = False
        
        # Store initial weights
        self.weight_history.append((self.perceptron.w.copy(), self.perceptron.b))
        
    def world_to_screen(self, x, y, plot_x, plot_y):
        """Convert world coordinates (-0.2 to 1.2) to screen coordinates"""
        screen_x = plot_x + (x + 0.2) * (PLOT_SIZE / 1.4)
        screen_y = plot_y + (1.2 - y) * (PLOT_SIZE / 1.4)
        return int(screen_x), int(screen_y)
    
    def draw_plot_background(self, plot_x, plot_y):
        """Draw the plot background and grid"""
        # Draw plot border
        pygame.draw.rect(self.screen, BLACK, (plot_x, plot_y, PLOT_SIZE, PLOT_SIZE), 2)
        
        # Draw grid
        for i in range(5):
            # Vertical lines
            x = plot_x + i * (PLOT_SIZE / 4)
            pygame.draw.line(self.screen, LIGHT_GRAY, (x, plot_y), (x, plot_y + PLOT_SIZE), 1)
            
            # Horizontal lines
            y = plot_y + i * (PLOT_SIZE / 4)
            pygame.draw.line(self.screen, LIGHT_GRAY, (plot_x, y), (plot_x + PLOT_SIZE, y), 1)
        
        # Draw axes labels
        for i in range(5):
            val = i * 0.35 - 0.2  # Map to -0.2 to 1.2 range
            label = self.small_font.render(f"{val:.1f}", True, BLACK)
            
            # X-axis labels
            x = plot_x + i * (PLOT_SIZE / 4) - 10
            self.screen.blit(label, (x, plot_y + PLOT_SIZE + 5))
            
            # Y-axis labels
            y = plot_y + PLOT_SIZE - i * (PLOT_SIZE / 4) - 10
            self.screen.blit(label, (plot_x - 30, y))
    
    def draw_data_points(self, plot_x, plot_y):
        """Draw the training data points"""
        for i, (point, label) in enumerate(zip(self.X, self.y)):
            screen_x, screen_y = self.world_to_screen(point[0], point[1], plot_x, plot_y)
            
            if label == 0:
                pygame.draw.circle(self.screen, RED, (screen_x, screen_y), 8)
                pygame.draw.circle(self.screen, BLACK, (screen_x, screen_y), 8, 2)
            else:
                # Draw X for class 1
                pygame.draw.line(self.screen, BLUE, (screen_x-6, screen_y-6), (screen_x+6, screen_y+6), 3)
                pygame.draw.line(self.screen, BLUE, (screen_x-6, screen_y+6), (screen_x+6, screen_y-6), 3)
    
    def draw_decision_boundary(self, plot_x, plot_y):
        """Draw the current decision boundary"""
        if np.allclose(self.perceptron.w, 0):
            return
            
        w1, w2 = self.perceptron.w
        b = self.perceptron.b
        
        if abs(w2) > 1e-6:  # w2 is not zero
            # Draw line: w1*x + w2*y + b = 0 => y = -(w1*x + b)/w2
            x_vals = np.linspace(-0.2, 1.2, 100)
            points = []
            
            for x in x_vals:
                y = -(w1 * x + b) / w2
                if -0.2 <= y <= 1.2:  # Only draw within plot bounds
                    screen_x, screen_y = self.world_to_screen(x, y, plot_x, plot_y)
                    points.append((screen_x, screen_y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, GREEN, False, points, 3)
        
        elif abs(w1) > 1e-6:  # w1 is not zero, w2 is zero (vertical line)
            x = -b / w1
            if -0.2 <= x <= 1.2:
                screen_x1, screen_y1 = self.world_to_screen(x, -0.2, plot_x, plot_y)
                screen_x2, screen_y2 = self.world_to_screen(x, 1.2, plot_x, plot_y)
                pygame.draw.line(self.screen, GREEN, (screen_x1, screen_y1), (screen_x2, screen_y2), 3)
    
    def draw_network_diagram(self, x, y, width, height):
        """Draw a simple neural network diagram"""
        # Input nodes
        input1_pos = (x + 50, y + height // 3)
        input2_pos = (x + 50, y + 2 * height // 3)
        
        # Output node
        output_pos = (x + width - 50, y + height // 2)
        
        # Draw connections with weight labels
        w1, w2 = self.perceptron.w
        b = self.perceptron.b
        
        # Connection from input1 to output
        pygame.draw.line(self.screen, BLACK, input1_pos, output_pos, 2)
        weight_text = self.small_font.render(f"w1: {w1:.2f}", True, BLACK)
        self.screen.blit(weight_text, (x + 80, y + 20))
        
        # Connection from input2 to output
        pygame.draw.line(self.screen, BLACK, input2_pos, output_pos, 2)
        weight_text = self.small_font.render(f"w2: {w2:.2f}", True, BLACK)
        self.screen.blit(weight_text, (x + 80, y + height - 40))
        
        # Draw nodes
        pygame.draw.circle(self.screen, BLUE, input1_pos, 20)
        pygame.draw.circle(self.screen, BLUE, input2_pos, 20)
        pygame.draw.circle(self.screen, RED, output_pos, 20)
        
        # Node labels
        input1_text = self.small_font.render("x1", True, WHITE)
        input2_text = self.small_font.render("x2", True, WHITE)
        output_text = self.small_font.render("y", True, WHITE)
        
        self.screen.blit(input1_text, (input1_pos[0] - 8, input1_pos[1] - 8))
        self.screen.blit(input2_text, (input2_pos[0] - 8, input2_pos[1] - 8))
        self.screen.blit(output_text, (output_pos[0] - 6, output_pos[1] - 8))
        
        # Bias
        bias_text = self.small_font.render(f"bias: {b:.2f}", True, BLACK)
        self.screen.blit(bias_text, (x + width // 2 - 30, y + height - 20))
    
    def draw_error_graph(self, x, y, width, height):
        """Draw the error over epochs graph"""
        if len(self.error_history) < 2:
            return
            
        # Draw graph border
        pygame.draw.rect(self.screen, BLACK, (x, y, width, height), 2)
        
        # Scale error values to fit in the graph
        max_error = max(self.error_history) if self.error_history else 1
        if max_error == 0:
            max_error = 1
            
        points = []
        for i, error in enumerate(self.error_history):
            graph_x = x + (i / max(len(self.error_history) - 1, 1)) * width
            graph_y = y + height - (error / max_error) * height
            points.append((graph_x, graph_y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, RED, False, points, 2)
        
        # Draw current epoch marker
        if self.epoch < len(self.error_history):
            current_x = x + (self.epoch / max(len(self.error_history) - 1, 1)) * width
            current_y = y + height - (self.error_history[self.epoch] / max_error) * height
            pygame.draw.circle(self.screen, BLUE, (int(current_x), int(current_y)), 5)
    
    def train_one_epoch(self):
        """Perform one epoch of training"""
        if self.training_complete:
            return
            
        errors = 0
        weight_update = np.zeros_like(self.perceptron.w)
        bias_update = 0.0
        
        for xi, yi in zip(self.X, self.y):
            yhat = self.perceptron.predict_single(xi)
            error = yi - yhat
            
            if error != 0:
                weight_update += self.perceptron.lr * error * xi
                bias_update += self.perceptron.lr * error
                errors += 1
        
        self.perceptron.w += weight_update
        self.perceptron.b += bias_update
        
        # Store history
        self.weight_history.append((self.perceptron.w.copy(), self.perceptron.b))
        self.error_history.append(errors)
        
        if errors == 0 or self.epoch >= 99:
            self.training_complete = True
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.setup_training()
                elif event.key == pygame.K_1:
                    self.current_gate = "AND"
                    self.setup_training()
                elif event.key == pygame.K_2:
                    self.current_gate = "OR"
                    self.setup_training()
                elif event.key == pygame.K_3:
                    self.current_gate = "NAND"
                    self.setup_training()
                elif event.key == pygame.K_4:
                    self.current_gate = "XOR"
                    self.setup_training()
        return True
    
    def draw_instructions(self):
        """Draw control instructions"""
        instructions = [
            "Controls:",
            "SPACE - Pause/Resume",
            "R - Restart",
            "1 - AND Gate",
            "2 - OR Gate", 
            "3 - NAND Gate",
            "4 - XOR Gate"
        ]
        
        y_offset = WINDOW_HEIGHT - 160
        for i, instruction in enumerate(instructions):
            color = BLACK if i == 0 else DARK_GRAY
            text = self.small_font.render(instruction, True, color)
            self.screen.blit(text, (20, y_offset + i * 20))
    
    def run(self):
        """Main animation loop"""
        running = True
        
        while running:
            running = self.handle_events()
            
            # Clear screen
            self.screen.fill(WHITE)
            
            # Update training
            if not self.paused and not self.training_complete:
                self.train_one_epoch()
                self.epoch += 1
            
            # Draw main plot
            plot_x, plot_y = 50, 50
            self.draw_plot_background(plot_x, plot_y)
            self.draw_data_points(plot_x, plot_y)
            self.draw_decision_boundary(plot_x, plot_y)
            
            # Draw network diagram
            network_x = PLOT_SIZE + 100
            network_y = 50
            self.draw_network_diagram(network_x, network_y, 300, 200)
            
            # Draw error graph
            error_x = 50
            error_y = PLOT_SIZE + 100
            self.draw_error_graph(error_x, error_y, 400, 150)
            
            # Draw title and status
            title = self.font.render(f"{self.current_gate} Gate Training - Epoch {self.epoch}", True, BLACK)
            self.screen.blit(title, (50, 10))
            
            status = "COMPLETED" if self.training_complete else ("PAUSED" if self.paused else "TRAINING")
            status_color = GREEN if self.training_complete else (RED if self.paused else BLUE)
            status_text = self.font.render(f"Status: {status}", True, status_color)
            self.screen.blit(status_text, (PLOT_SIZE + 100, 10))
            
            # Draw current error
            current_error = self.error_history[-1] if self.error_history else 0
            error_text = self.small_font.render(f"Current Errors: {current_error}", True, BLACK)
            self.screen.blit(error_text, (50, PLOT_SIZE + 80))
            
            # Draw accuracy
            predictions = self.perceptron.predict(self.X)
            accuracy = np.mean(predictions == self.y) * 100
            accuracy_text = self.small_font.render(f"Accuracy: {accuracy:.1f}%", True, BLACK)
            self.screen.blit(accuracy_text, (200, PLOT_SIZE + 80))
            
            # Draw legend
            legend_x = PLOT_SIZE + 100
            legend_y = 280
            legend_text = self.small_font.render("Legend:", True, BLACK)
            self.screen.blit(legend_text, (legend_x, legend_y))
            
            # Class 0 (red circles)
            pygame.draw.circle(self.screen, RED, (legend_x + 20, legend_y + 25), 8)
            pygame.draw.circle(self.screen, BLACK, (legend_x + 20, legend_y + 25), 8, 2)
            class0_text = self.small_font.render("Class 0", True, BLACK)
            self.screen.blit(class0_text, (legend_x + 35, legend_y + 20))
            
            # Class 1 (blue X)
            pygame.draw.line(self.screen, BLUE, (legend_x + 14, legend_y + 44), (legend_x + 26, legend_y + 56), 3)
            pygame.draw.line(self.screen, BLUE, (legend_x + 14, legend_y + 56), (legend_x + 26, legend_y + 44), 3)
            class1_text = self.small_font.render("Class 1", True, BLACK)
            self.screen.blit(class1_text, (legend_x + 35, legend_y + 45))
            
            # Decision boundary
            pygame.draw.line(self.screen, GREEN, (legend_x + 10, legend_y + 75), (legend_x + 30, legend_y + 75), 3)
            boundary_text = self.small_font.render("Decision Boundary", True, BLACK)
            self.screen.blit(boundary_text, (legend_x + 35, legend_y + 70))
            
            # Draw instructions
            self.draw_instructions()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    animator = NeuralNetworkAnimator()
    animator.run()
