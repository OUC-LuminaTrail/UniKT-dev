"""模型包。

导入本包只会触发**静态注册发现**:扫描 ``model/`` 下源码(不导入任何模型代码),把所有
``@register_trainer`` / ``@register_model_config`` / ``@register_analyzer`` 注册项写入对应
注册表的懒索引。模型代码在 ``TRAINERS.get(...)`` 等调用时才按需导入(懒加载,多环境安全)。
"""

from pathlib import Path

from utils.core import discover_registrations

discover_registrations(Path(__file__).parent, "model")
