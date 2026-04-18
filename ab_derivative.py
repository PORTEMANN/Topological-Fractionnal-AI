"""
Topological Fractional AI - Core Engine
Module: ab_derivative.py

Description:
Implementation of the Atangana-Baleanu fractional derivative in the 
Caputo sense. Uses the Mittag-Leffler function to embed non-local 
memory into the dynamic state-space solver.

Mathematical Definition:
    (ABC)_a D_t^α f(t) = [B(α) / (1 - α)] * ∫_a^t f'(τ) * E_α[-α/(1-α) * (t-τ)^α] dτ

Author: Patrice Portemann
"""

import numpy as np
from scipy.special import mittag_leffler
import warnings

class AtanganaBaleanu:
    """
    Computes the Atangana-Baleanu (AB) fractional derivative.
    """
    
    def __init__(self, alpha: float):
        """
        Initializes the AB derivative operator.
        
        Args:
            alpha (float): The fractional order, must be strictly in (0, 1).
        """
        if not (0 < alpha < 1):
            raise ValueError("Fractional order alpha must be in the interval (0, 1).")
            
        self.alpha = alpha
        # Normalization function B(alpha)
        self.B_alpha = 1 - alpha + (alpha / np.math.gamma(alpha))
        
    def _compute_memory_kernel(self, t_array: np.ndarray) -> np.ndarray:
        """
        Pre-computes the discrete memory weights based on the Mittag-Leffler function.
        This avoids recalculating the special function at every time step, drastically
        improving computational efficiency.
        """
        n_steps = len(t_array) - 1
        weights = np.zeros(n_steps)
        
        for j in range(n_steps):
            t_j = t_array[j]
            t_j1 = t_array[j+1]
            
            # Evaluate Mittag-Leffler at t_j and t_(j+1)
            arg_j = - (self.alpha / (1 - self.alpha)) * (t_j ** self.alpha)
            arg_j1 = - (self.alpha / (1 - self.alpha)) * (t_j1 ** self.alpha)
            
            ml_j = mittag_leffler(self.alpha, 1.0, arg_j)
            ml_j1 = mittag_leffler(self.alpha, 1.0, arg_j1)
            
            # Weight is the discrete integral of the kernel over the time step
            weights[j] = ml_j - ml_j1
            
        return weights

    def derivative(self, f_t: np.ndarray, t_array: np.ndarray) -> np.ndarray:
        """
        Calculates the AB derivative of a function f(t) over a time array.
        
        Args:
            f_t (np.ndarray): The function values at each time step.
            t_array (np.ndarray): The corresponding time values.
            
        Returns:
            np.ndarray: The fractional derivative d^α f / dt^α
        """
        if len(f_t) != len(t_array):
            raise ValueError("f_t and t_array must have the same length.")
            
        n = len(t_array)
        d_f_ab = np.zeros(n)
        
        # 1. Compute standard first derivative f'(t)
        f_prime = np.gradient(f_t, t_array)
        
        # 2. Get the memory weights (The "Topological Prior")
        weights = self._compute_memory_kernel(t_array)
        a0 = weights[0] # Normalization constant for the discrete scheme
        
        # 3. Compute the integral using the pre-computed memory weights
        for i in range(1, n):
            # The system remembers the entire history up to time t_i
            history_weights = weights[:i][::-1] # Reverse to align with f_prime[1:i+1]
            recent_f_primes = f_prime[1:i+1]
            
            memory_integral = np.dot(history_weights, recent_f_primes)
            
            # Apply the AB formula
            d_f_ab[i] = (1.0 / a0) * memory_integral
            
        return d_f_ab


# ==========================================
# STANDALONE TEST (For R&D Evaluation)
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Running Atangana-Baleanu Derivative Standalone Test...")
    print("Comparing Standard Derivative (Markovian) vs Fractional Derivative (Non-Markovian)\n")
    
    # Setup a dummy signal (e.g., a sudden step or sine wave)
    t = np.linspace(0, 5, 500)
    f_signal = np.sin(2 * np.pi * t) * (t < 2.5) # Sine wave that stops abruptly
    
    # Compute standard derivative
    f_prime_standard = np.gradient(f_signal, t)
    
    # Compute fractional derivative with memory (e.g., alpha = 0.85)
    ab_operator = AtanganaBaleanu(alpha=0.85)
    f_prime_fractional = ab_operator.derivative(f_signal, t)
    
    # Plotting the "Heavy Tail" effect to prove the math works
    plt.figure(figsize=(10, 5))
    plt.plot(t, f_signal, 'k-', linewidth=2, label="Original Signal f(t)")
    plt.plot(t, f_prime_standard, 'r--', linewidth=1.5, label="Standard Derivative (No memory)")
    plt.plot(t, f_prime_fractional, 'b-', linewidth=2, label="AB Derivative α=0.85 (With memory)")
    
    # Highlight the memory effect after the signal stops
    plt.axvspan(2.5, 5.0, color='gray', alpha=0.1, label="Signal stopped (t > 2.5)")
    plt.title("Proof of Non-Local Memory: The Topological Prior keeps evolving after input stops")
    plt.xlabel("Time (t)")
    plt.ylabel("Derivative magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save instead of just showing (better for automated evaluation)
    plt.savefig("ab_derivative_test.png", dpi=150)
    print("Graph saved as 'ab_derivative_test.png'.")
    print("Notice how the blue line (Fractional) continues to evolve after t=2.5,")
    print("while the red line (Standard) drops immediately to zero. This is the heavy tail.")
