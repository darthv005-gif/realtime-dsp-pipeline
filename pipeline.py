import time
import queue
import threading
import numpy as np
from scipy.signal import firwin, lfilter

class RealTimeDSPPipeline:
    def __init__(self, sample_rate=44100, initial_chunk_size=1024, max_queue_size=16):
        self.sample_rate = sample_rate
        self.chunk_size = initial_chunk_size
        self.max_queue_size = max_queue_size
        
        # Thread-safe bounded queue and synchronization lock for atomic drop-head operations
        self.queue_lock = threading.Lock()
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        
        self.is_running = False
        self.latency_history = []
        
        # Design a linear-phase FIR bandpass filter (Hamming window)
        # E.g., cutoff frequencies normalized to Nyquist
        self.numtaps = 101
        self.filter_coeffs = firwin(self.numtaps, [0.1, 0.4], window='hamming', pass_zero=False)
        self.filter_state = np.zeros(self.numtaps - 1)

    def producer_simulation(self, duration_seconds=5):
        """Simulates an incoming high-frequency data/audio stream."""
        start_time = time.time()
        sequence_id = 0
        
        while self.is_running and (time.time() - start_time) < duration_seconds:
            # Generate dummy incoming raw audio/sensor signal block
            raw_chunk = np.sin(2 * np.pi * 1000 * np.arange(self.chunk_size) / self.sample_rate) \
                        + 0.5 * np.random.randn(self.chunk_size)
            
            packet = {
                'sequence_id': sequence_id,
                'timestamp': time.time(),
                'data': raw_chunk
            }
            
            # Atomic Drop-Head (Drop-Oldest) Backpressure Implementation
            with self.queue_lock:
                if self.input_queue.full():
                    try:
                        discarded = self.input_queue.get_nowait()
                        print(f"[Warning] Queue saturation! Dropped stale frame ID: {discarded['sequence_id']}")
                    except queue.Empty:
                        pass
                
                self.input_queue.put(packet)
            
            sequence_id += 1
            time.sleep(0.015) # Simulating incoming stream rate

    def consumer_worker(self):
        """Processes chunks, applies DSP filters, monitors latency, and adapts buffering."""
        while self.is_running:
            try:
                packet = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            t_start = time.time()
            
            # DSP Processing: FIR Filtering + FFT Spectral Analysis
            filtered_chunk, self.filter_state = lfilter(
                self.filter_coeffs, 1.0, packet['data'], zi=self.filter_state
            )
            spectrum = np.fft.rfft(filtered_chunk)
            
            processing_time = time.time() - t_start
            self._adapt_parameters(processing_time)
            
            self.input_queue.task_done()

    def _adapt_parameters(self, processing_time):
        """Dynamically adjusts chunk size based on rolling latency metrics."""
        self.latency_history.append(processing_time)
        if len(self.latency_history) > 20:
            self.latency_history.pop(0)
            avg_latency = sum(self.latency_history) / len(self.latency_history)
            
            # If processing takes too long (>20ms), scale down chunk size to reduce CPU load
            if avg_latency > 0.020 and self.chunk_size > 512:
                self.chunk_size = max(512, self.chunk_size // 2)
                print(f"[Adaptation] High latency ({avg_latency*1000:.1f}ms). Reduced chunk size to {self.chunk_size}")
            # If underutilized (<5ms), scale up for better frequency resolution
            elif avg_latency < 0.005 and self.chunk_size < 4096:
                self.chunk_size = min(4096, self.chunk_size * 2)
                print(f"[Adaptation] Low latency ({avg_latency*1000:.1f}ms). Increased chunk size to {self.chunk_size}")

    def run(self):
        print("Starting Real-Time DSP Pipeline...")
        self.is_running = True
        
        producer_thread = threading.Thread(target=self.producer_simulation)
        consumer_thread = threading.Thread(target=self.consumer_worker)
        
        producer_thread.start()
        consumer_thread.start()
        
        producer_thread.join()
        self.is_running = False
        consumer_thread.join(timeout=1.0)
        print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    pipeline = RealTimeDSPPipeline()
    pipeline.run()
