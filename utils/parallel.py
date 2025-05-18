import concurrent.futures
from utils.logger import logger

class ParallelProcessor:
    def __init__(self, max_workers=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    def submit_task(self, func, *args, **kwargs):
        return self.executor.submit(func, *args, **kwargs)
    
    def run_tasks(self, tasks):
        futures = []
        results = []
        
        for func, args, kwargs in tasks:
            futures.append(self.executor.submit(func, *args, **kwargs))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Task execution error: {str(e)}")
                results.append(None)
        
        return results
