"""Lock serializing DATA_SOURCES clear+rediscover (refresh) against readers.

A refresh clears and rebuilds the DATA_SOURCES index; without serialization a
concurrent /api/datasets reader could observe an empty registry mid-scan and
raise a spurious 500. The lock makes clear+rediscover atomic w.r.t. readers.
"""

import threading

registry_lock = threading.RLock()
