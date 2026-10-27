"""Honor Ctrl+C after a complete PPO update, then save its consistent state."""
from pathlib import Path
import signal
from types import MethodType


class TrainingStopped(Exception):
    pass


class IterationStop:
    def __init__(self, runner, log_dir):
        self.runner = runner
        self.log_dir = Path(log_dir)
        self.requested = False
        self.saved_path = None

    def request(self, signum=None, frame=None):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print('[INFO] Stop requested; finishing this PPO iteration and saving. Press Ctrl+C again to force exit.',flush=True)

    def __enter__(self):
        self.previous_signal = signal.getsignal(signal.SIGINT)
        self.previous_log = self.runner.logger.log
        signal.signal(signal.SIGINT,self.request)
        control = self
        def log(logger,*args,**kwargs):
            control.previous_log(*args,**kwargs)
            if control.requested:
                # OnPolicyRunner sets current_learning_iteration after update
                # and before logger.log. No partial optimizer step is saved.
                control.saved_path = control.log_dir/f'model_{control.runner.current_learning_iteration}.pt'
                control.runner.save(str(control.saved_path))
                writer = getattr(logger,'writer',None)
                if writer is not None: writer.flush()
                print(f'[INFO] Saved interrupted training: {control.saved_path}',flush=True)
                raise TrainingStopped
        self.runner.logger.log = MethodType(log,self.runner.logger)
        return self

    def __exit__(self,kind,value,traceback):
        signal.signal(signal.SIGINT,self.previous_signal)
        self.runner.logger.log = self.previous_log
        return kind is TrainingStopped
