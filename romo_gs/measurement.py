"""Separate process RSS high-water and sampled MPS allocation statistics.

On Apple unified memory these quantities overlap and MUST NOT be added.
MPS peaks are 2ms polling observations, not exact hardware peak counters.
"""
import platform
import resource
import threading
import time
import torch


def sync(device):
    if str(device)=='mps':torch.mps.synchronize()


class MemoryMonitor:
    def __init__(self,device):
        self.device=str(device);self.stop_event=threading.Event();self.samples=[]
    def sample(self):
        live=driver=0
        if self.device=='mps':
            live=torch.mps.current_allocated_memory();driver=torch.mps.driver_allocated_memory()
        self.samples.append((time.perf_counter(),live,driver))
    def __enter__(self):
        self.sample();self.baseline=self.samples[0]
        if self.device=='mps':
            def poll():
                while not self.stop_event.wait(.002):self.sample()
            self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()
        return self
    def __exit__(self,*args):
        self.sample();self.stop_event.set()
        if self.device=='mps':self.thread.join()
    def result(self):
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system()!='Darwin':rss*=1024
        return {'process_peak_rss_bytes':rss,'mps_baseline_live_bytes':self.baseline[1],
                'mps_sampled_peak_live_bytes':max(x[1] for x in self.samples),
                'mps_sampled_peak_driver_bytes':max(x[2] for x in self.samples),
                'mps_sampled_peak_increment_bytes':max(x[1] for x in self.samples)-self.baseline[1],
                'mps_samples':len(self.samples),'sampling_period_seconds':.002,
                'measurement_note':'RSS high-water since process start; MPS live/driver sampled at2ms; overlap in unified memory, do not sum'}
