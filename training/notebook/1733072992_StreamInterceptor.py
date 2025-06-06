import sys
import threading
import logging


class StreamInterceptor:
    def __init__(self, real_stdout, local_hash, key="txt_in", cache_size=16, backup_timeout=5.0):
        """
        Initialize the StreamInterceptor.

        Args:
            real_stdout: The original stdout stream.
            local_hash: The local state hash for storing text data.
            key (str): The key in the local hash where text data will be stored.
            cache_size (int): Maximum size of the cache before flushing.
            backup_timeout (float): Time (in seconds) to wait for Console to pick up data before backup dump.
        """
        self.real_stdout = real_stdout
        self.local_hash = local_hash
        self.key = key
        self.cache_size = cache_size
        self.backup_timeout = backup_timeout

        self.cache = []
        self.lock = threading.Lock()
        self.last_flush_time = threading.current_thread().ident

    def write(self, message):
        """
        Write a message to the cache and flush if the cache is full.
        """
        with self.lock:
            self.cache.append(message)
            if sum(len(m) for m in self.cache) >= self.cache_size:
                self.flush_to_local_hash()

    def flush(self):
        """
        Flush any remaining cache to the local state hash and the real stdout.
        """
        with self.lock:
            if self.cache:
                self.flush_to_local_hash()
            self.real_stdout.flush()

    def flush_to_local_hash(self):
        """
        Flush the cache to the local hash.
        If the local hash is unresponsive, dump the cache to the logger and real stdout.
        """
        if self.key not in self.local_hash:
            self.local_hash[self.key] = []

        try:
            # Check if the console plugin is responsive
            self.local_hash[self.key].extend(self.cache)
            if len(self.local_hash[self.key]) > self.cache_size:
                raise ValueError("Local hash backup queue exceeded capacity")
            self.cache.clear()

        except Exception as e:
            # Log the error and flush to real stdout as a backup
            logging.error(
                f"Error flushing to local hash ({self.key}): {e}. Dumping to logger and stdout."
            )
            self.real_stdout.write("".join(self.cache))
            self.real_stdout.flush()
            self.cache.clear()

    def isatty(self):
        """
        Return whether the original stdout is a TTY (for compatibility).
        """
        return self.real_stdout.isatty()

    def fileno(self):
        """
        Return the file descriptor of the original stdout (for compatibility).
        """
        return self.real_stdout.fileno()
