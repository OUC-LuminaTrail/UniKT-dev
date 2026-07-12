# 数据处理

模型级数据基类与数据源基类。各数据集处理器（`utils/data_process/<dataset>.py`）均继承 `DataSource`，按需实现 `load_src_data` / `clean_raw_data` / `transform_data`。

```{eval-rst}
.. automodule:: utils.model_data.base_model_data
   :members:
```

```{eval-rst}
.. automodule:: utils.model_data.question_model_data
   :members:
```

```{eval-rst}
.. automodule:: utils.model_data.skill_model_data
   :members:
```

```{eval-rst}
.. automodule:: utils.data_process.data_source
   :members:
```
